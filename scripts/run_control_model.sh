#!/usr/bin/env bash
set -euo pipefail
python -m src.models.train_crossval \
    --config configs/control_model.yaml \
    --experiment-name control
python -m src.utils.summarize_results --results-dir results/control
