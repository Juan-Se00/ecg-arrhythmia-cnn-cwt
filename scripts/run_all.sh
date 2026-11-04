#!/usr/bin/env bash
set -euo pipefail
echo "=== 1/6 Preprocessing ===" && bash scripts/run_preprocessing.sh
echo "=== 2/6 Balancing ==="      && bash scripts/run_balance.sh
echo "=== 3/6 Hyperparameter search ===" && bash scripts/run_hyperparam_search.sh
echo "=== 4/6 Control model ===" && bash scripts/run_control_model.sh
echo "=== 5/6 Main model ==="     && bash scripts/run_main_model.sh
echo "=== 6/6 Compare ==="        && python -m src.utils.compare_models
echo "Done."
