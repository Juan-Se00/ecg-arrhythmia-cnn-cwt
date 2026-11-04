#!/usr/bin/env bash
set -euo pipefail
python -m src.models.train_crossval \
    --config configs/main_model.yaml \
    --experiment-name main
python -m src.utils.summarize_results --results-dir results/main
