#!/usr/bin/env bash
# SignSight one-shot environment + dataset setup
# Usage: ./prepare.sh [--skip-dataset] [--skip-unsw] [--python PYTHON_BIN]

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_DATASET=false
SKIP_UNSW=false

for arg in "$@"; do
    case "$arg" in
        --skip-dataset) SKIP_DATASET=true ;;
        --skip-unsw)    SKIP_UNSW=true ;;
        --python=*)     PYTHON_BIN="${arg#--python=}" ;;
        --python)       PYTHON_BIN="$2"; shift ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "=== SignSight environment bootstrap ==="
echo "Project root: ${PROJECT_ROOT}"
echo "Python:       ${PYTHON_BIN}"
echo "Venv:         ${VENV_DIR}"
echo

# ---- 1. Python venv ----------------------------------------------------
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[*] Creating virtual environment..."
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
    echo "[*] Virtual environment already exists"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

echo "[*] Upgrading pip/setuptools/wheel..."
pip install -q --upgrade pip setuptools wheel

# ---- 2. Dependencies ---------------------------------------------------
echo "[*] Installing runtime dependencies from requirements.txt..."
pip install -q -r "${PROJECT_ROOT}/requirements.txt"

# Optional: install test deps if pytest not present
if ! python -c "import pytest" 2>/dev/null; then
    echo "[*] Installing test dependencies (pytest)..."
    pip install -q pytest
fi

# ---- 3. Dataset --------------------------------------------------------
if [[ "$SKIP_DATASET" == false ]]; then
    echo
    echo "[*] Setting up dataset..."
    if [[ "$SKIP_UNSW" == true ]]; then
        "${PROJECT_ROOT}/datasets/data.sh" --skip-unsw
    else
        "${PROJECT_ROOT}/datasets/data.sh"
    fi
else
    echo
    echo "[*] Skipping dataset download (--skip-dataset)"
fi

# ---- 4. Smoke test -----------------------------------------------------
echo
echo "[*] Running smoke test..."
python -c "
from model import HybridNIDSModel
from model.config import NIDSConfig, IsolationForestConfig, LightGBMConfig
print('  Imports OK')
m = HybridNIDSModel(NIDSConfig(
    stage1=IsolationForestConfig(n_estimators=10),
    stage2=LightGBMConfig(n_estimators=10),
    build_attack_family_head=False
))
print('  Model construction OK')
print('  All checks passed!')
"

# ---- 5. Summary --------------------------------------------------------
echo
echo "=== Setup complete ==="
echo
echo "Activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Quick commands:"
echo "  # Full LODO evaluation (~10 min)"
echo "  python -m model.train"
echo
echo "  # Zero-day holdout (train Mon-Thu, test Fri)"
echo "  python -m model.train --strategy holdout"
echo
echo "  # Run unit tests"
echo "  python -m pytest tests/test_model.py -v"
echo
echo "Artifacts will be written to: ${PROJECT_ROOT}/model/artifacts/"