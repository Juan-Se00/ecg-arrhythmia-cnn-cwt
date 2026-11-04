#!/usr/bin/env bash
set -euo pipefail
python -m src.models.search_hyperparameters \
    --config configs/custom_cnn.yaml \
    --trials 6 \
    "$@"
python -m src.utils.extract_best_params \
    --checkpoint models/best_overall_loss_model.pth \
    --output configs/best_params.json
