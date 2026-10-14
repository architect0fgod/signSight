"""Ablation: Hybrid (IF + LGBM) without subnet features (holdout: Mon-Thu -> Fri)."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import DEFAULT_DATA_DIR, NIDSConfig, DAY_FILES
from .data import FlowPreprocessor, load_days, normalise_labels
from .thresholds import select_threshold
import model.data as data_module


def run_hybrid_ablation_no_subnet():
    cfg = NIDSConfig()
    
    train_days = ["mon", "tue", "wed", "thu"]
    test_days = ["fri"]
    
    print(f"[*] Loading train days: {train_days}")
    frame_tr = load_days(train_days, cfg.data_dir)
    print(f"[*] Loading test days: {test_days}")
    frame_te = load_days(test_days, cfg.data_dir)
    
    y_tr = normalise_labels(frame_tr["Label"], cfg.include_attempted).to_numpy()
    y_te = normalise_labels(frame_te["Label"], cfg.include_attempted).to_numpy()
    
    print(f"[*] Train: {len(frame_tr):,} flows ({int(y_tr.sum()):,} attack)")
    print(f"[*] Test:  {len(frame_te):,} flows ({int(y_te.sum()):,} attack)")
    
    # Fit preprocessor WITHOUT subnet features
    original_subnet_columns = data_module._subnet_columns
    data_module._subnet_columns = lambda df: []
    
    pre = FlowPreprocessor()
    pre.fit(frame_tr)
    
    # Restore
    data_module._subnet_columns = original_subnet_columns
    
    # Stage 1: Isolation Forest on benign only
    benign_mask = y_tr == 0
    x_tr_stage1 = pre.transform(frame_tr, impute=True)
    
    print("[*] Training Isolation Forest...")
    start = time.time()
    iso = IsolationForest(
        n_estimators=cfg.stage1.n_estimators,
        max_samples=cfg.stage1.max_samples,
        contamination=cfg.stage1.contamination,
        max_features=cfg.stage1.max_features,
        bootstrap=cfg.stage1.bootstrap,
        random_state=cfg.stage1.random_state,
        n_jobs=cfg.stage1.n_jobs,
    )
    iso.fit(x_tr_stage1.to_numpy()[benign_mask])
    print(f"    done in {time.time() - start:.1f}s")
    
    # Get anomaly scores
    x_te_stage1 = pre.transform(frame_te, impute=True)
    anomaly_te = -iso.score_samples(x_te_stage1.to_numpy())
    novelty_cut = float(np.quantile(-iso.score_samples(x_tr_stage1.to_numpy()[benign_mask]), 1.0 - cfg.stage1.contamination))
    
    # Stage 2: LightGBM on flows + anomaly
    x_tr_stage2 = pre.transform(frame_tr, impute=False)
    x_tr_stage2["iso_anomaly_score"] = -iso.score_samples(x_tr_stage1.to_numpy())
    
    x_te_stage2 = pre.transform(frame_te, impute=False)
    x_te_stage2["iso_anomaly_score"] = anomaly_te
    
    print(f"[*] Features (no subnet + anomaly): {x_tr_stage2.shape[1]}")
    
    clf = LGBMClassifier(objective="binary", **cfg.stage2.as_params())
    
    print("[*] Training LightGBM...")
    start = time.time()
    clf.fit(x_tr_stage2, y_tr)
    print(f"    done in {time.time() - start:.1f}s")
    
    proba_te = clf.predict_proba(x_te_stage2)[:, list(clf.classes_).index(1)]
    
    # Threshold selection
    op = select_threshold(y_te, proba_te, max_fpr=cfg.max_fpr, confidence=cfg.fpr_confidence)
    pred = (proba_te >= op.threshold).astype(np.int8)
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    
    print("\n" + "=" * 70)
    print("  ABLATION: HYBRID (IF + LGBM, NO SUBNET) - HOLDOUT FRI")
    print("=" * 70)
    print(f"  FPR budget          : {cfg.max_fpr:.4%}")
    print(f"  Threshold           : {op.threshold:.6f}")
    print(f"  FPR observed        : {fpr:.5%}")
    print(f"  FPR 95% UB          : {op.fpr_upper_bound:.5%}")
    print(f"  Recall @ budget     : {recall:.4%}")
    print(f"  Precision @ budget  : {precision:.4%}")
    print(f"  ROC-AUC             : {roc_auc_score(y_te, proba_te):.4f}")
    print(f"  PR-AUC              : {average_precision_score(y_te, proba_te):.4f}")
    print("-" * 70)
    print(classification_report(y_te, pred, target_names=["BENIGN", "ATTACK"], digits=4))
    
    # Per-family breakdown
    labels = frame_te["Label"].astype(str).str.strip()
    print("\n  Per-family recall @ threshold:")
    for label in sorted(set(labels[labels != "BENIGN"])):
        mask = (labels == label).to_numpy()
        n = int(mask.sum())
        fam_recall = np.mean(proba_te[mask] >= op.threshold)
        mean_prob = proba_te[mask].mean()
        print(f"    {label:<30} n={n:>8,}  recall={fam_recall:.4f}  mean_prob={mean_prob:.4f}")
    
    # Feature importance - check anomaly score share
    print("\n  Top 15 features (gain):")
    imp = pd.DataFrame({
        "feature": list(clf.feature_name_),
        "gain": clf.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    total_gain = imp["gain"].sum()
    imp["gain_share"] = imp["gain"] / total_gain
    for _, row in imp.head(15).iterrows():
        stage = "STAGE1" if row['feature'] == "iso_anomaly_score" else "flow"
        print(f"    {row['feature']:<35} gain_share={row['gain_share']:.4%}  [{stage}]")
    
    return {
        "threshold": float(op.threshold),
        "fpr": float(fpr),
        "fpr_ub": float(op.fpr_upper_bound),
        "recall": float(recall),
        "precision": float(precision),
        "roc_auc": float(roc_auc_score(y_te, proba_te)),
        "pr_auc": float(average_precision_score(y_te, proba_te)),
    }


if __name__ == "__main__":
    run_hybrid_ablation_no_subnet()