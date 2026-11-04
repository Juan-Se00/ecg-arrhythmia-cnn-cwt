#!/usr/bin/env bash
set -euo pipefail
python -m src.balancing.augment_and_balance \
    --data-root data/processed/ECG_spectrograms \
    --target 75340
