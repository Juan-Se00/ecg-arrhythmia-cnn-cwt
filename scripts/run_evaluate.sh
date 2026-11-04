#!/usr/bin/env bash
set -euo pipefail
python -m src.models.evaluate_detailed \
    --config configs/main_model.yaml \
    --arch ECGCNN \
    --params configs/best_params.json \
    --model-path models/best_overall_loss_model.pth \
    --output results/main/eval
