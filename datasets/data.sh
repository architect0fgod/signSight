#!/usr/bin/env bash
# Download and extract CICIDS2017-improved dataset for SignSight
# Usage: ./data.sh [--skip-unsw]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${SCRIPT_DIR}/cicimp"
ZIP_FILE="${DATASET_DIR}/CICIDS2017_improved.zip"

SKIP_UNSW=false
for arg in "$@"; do
    case "$arg" in
        --skip-unsw) SKIP_UNSW=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

mkdir -p "$DATASET_DIR"

echo "[*] Downloading CICIDS2017-improved from DistriNet..."
curl -L -o "$ZIP_FILE" \
  "https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/CICIDS2017_improved.zip"

echo "[*] Extracting dataset..."
unzip -o "$ZIP_FILE" -d "$DATASET_DIR"

# The zip contains Monday.csv ... Friday.csv; rename to lowercase
for day in Monday Tuesday Wednesday Thursday Friday; do
    if [[ -f "${DATASET_DIR}/${day}.csv" ]]; then
        mv "${DATASET_DIR}/${day}.csv" "${DATASET_DIR}/${day,,}.csv"
    fi
done

rm -f "$ZIP_FILE"

if [[ "$SKIP_UNSW" == false ]]; then
    echo "[*] Downloading UNSW-NB15 (optional supplementary data)..."
    curl -L -o "${DATASET_DIR}/UNSW_NB15_testing-set.csv" \
      "https://raw.githubusercontent.com/ushukkla/nospammers/master/UNSW_NB15_testing-set.csv" || true
    curl -L -o "${DATASET_DIR}/UNSW_NB15_training-set.csv" \
      "https://raw.githubusercontent.com/ushukkla/nospammers/master/UNSW_NB15_training-set.csv" || true
fi

echo "[*] Verifying expected CIC files..."
for f in monday.csv tuesday.csv wednesday.csv thursday.csv friday.csv; do
    if [[ ! -f "${DATASET_DIR}/${f}" ]]; then
        echo "[!] Missing ${f}" >&2
        exit 1
    fi
    lines=$(wc -l < "${DATASET_DIR}/${f}")
    echo "    ${f}: ${lines} lines"
done

echo "[*] Dataset ready in ${DATASET_DIR}"
