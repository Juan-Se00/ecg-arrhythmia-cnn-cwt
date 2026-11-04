.PHONY: help install preprocess balance search train-main train-control evaluate compare test clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:            ## Install package in editable mode
	pip install -e ".[dev]"

preprocess:         ## Download MIT-BIH and build spectrograms
	python -m src.data.build_dataset

balance:            ## Augment minority classes to 150,680 / class
	python -m src.balancing.augment_and_balance --data-root data/processed/ECG_spectrograms --target 75340

search:             ## Random hyperparameter search (6 trials)
	python -m src.models.search_hyperparameters --config configs/custom_cnn.yaml --trials 6

train-main:         ## Train the balanced model (5-fold CV)
	python -m src.models.train_crossval --config configs/main_model.yaml --experiment-name main

train-control:      ## Train the unbalanced model (5-fold CV)
	python -m src.models.train_crossval --config configs/control_model.yaml --experiment-name control

evaluate:           ## Evaluate a saved checkpoint
	python -m src.models.evaluate_detailed --config configs/main_model.yaml \
		--arch ECGCNN --params configs/best_params.json \
		--model-path models/best_overall_loss_model.pth \
		--output results/main/eval

compare:            ## Generate Table XI
	python -m src.utils.compare_models

test:               ## Run test suite
	pytest

clean:              ## Remove python caches
	rm -rf **/__pycache__ **/*.pyc .pytest_cache
