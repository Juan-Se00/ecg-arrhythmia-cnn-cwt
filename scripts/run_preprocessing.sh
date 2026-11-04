#!/usr/bin/env bash
set -euo pipefail
python -m src.data.build_dataset \
    --raw-dir data/raw \
    --out-dir data/processed/ECG_spectrograms \
    --degree 6
