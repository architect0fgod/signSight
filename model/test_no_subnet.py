"""Ablation: LightGBM without subnet features (holdout: Mon-Thu -> Fri)."""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .config import DEFAULT_DATA_DIR, NIDSConfig, DAY_FILES
from .data import FlowPreprocessor, load_days, normalise_labels
from .thresholds import select_threshold


def run_ablation_no_subnet():
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
    
    # Fit preprocessor WITHOUT subnet features (monkey-patch module function)
    import model.data as data_module
    original_subnet_columns = data_module._subnet_columns
    data_module._subnet_columns = lambda df: []
    
    pre = FlowPreprocessor()
    pre.fit(frame_tr)
    
    # Restore
    data_module._subnet_columns = original_subnet_columns
    
    x_tr = pre.transform(frame_tr, impute=False)
    x_te = pre.transform(frame_te, impute=False)
    
    print(f"[*] Features (no subnet): {x_tr.shape[1]}")
    print(f"[*] Feature names: {list(x_tr.columns)}")
    
    clf = LGBMClassifier(objective="binary", **cfg.stage2.as_params())
    
    print("[*] Training LightGBM...")
    start = time.time()
    clf.fit(x_tr, y_tr)
    print(f"    done in {time.time() - start:.1f}s")
    
    proba_te = clf.predict_proba(x_te)[:, list(clf.classes_).index(1)]
    
    # Threshold selection
    op = select_threshold(y_te, proba_te, max_fpr=cfg.max_fpr, confidence=cfg.fpr_confidence)
    pred = (proba_te >= op.threshold).astype(np.int8)
    
    # Metrics
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn) if tp + fn else 0
    precision = tp / (tp + fp) if tp + fp else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    
    print("\n" + "=" * 70)
    print("  ABLATION: LGBM ONLY (NO SUBNET FEATURES) - HOLDOUT FRI")
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
    
    # Feature importance
    print("\n  Top 15 features (gain):")
    imp = pd.DataFrame({
        "feature": list(clf.feature_name_),
        "gain": clf.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    total_gain = imp["gain"].sum()
    imp["gain_share"] = imp["gain"] / total_gain
    for _, row in imp.head(15).iterrows():
        print(f"    {row['feature']:<35} gain_share={row['gain_share']:.4%}")
    
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
    run_ablation_no_subnet()