# Multiclass ECG Arrhythmia Classification via CWT Spectro-Temporal Representations and 2D CNNs

Reference implementation for the paper:

> **Ardila Parra, J.S.** & Orjuela Cañón, A.D. (2024).
> *Multiclass Classification of Cardiac Arrhythmias Using Two-Dimensional
> Convolutional Neural Networks and ECG Spectro-Temporal Representations.*
> Undergraduate thesis, Universidad del Rosario & Escuela Colombiana de
> Ingeniería Julio Garavito.
> [📄 Read the full article (PDF)](reports/paper/Ardila_Parra_2024_ECG_Arrhythmia_Classification.pdf)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Reproduce paper tables

| Paper artifact | Command | Output |
|---|---|---|
| **Figure 1** | `jupyter nbconvert --execute notebooks/02_cwt_visualization.ipynb` | `reports/figures/figure1_spectrograms.png` |
| **Table III** | `python -m src.models.train_mobilenet --config configs/mobilenetv2.yaml --experiment fusion` | `results/mobilenet/*/final_results/` |
| **Table IV** | `bash scripts/run_balance.sh` then `ls data/processed/ECG_spectrograms/MLII/*/` | folder counts |
| **Table VI / VII / VIII** | `bash scripts/run_main_model.sh` | `results/main/` |
| **Figure 2** | (produced by Table VI) | `results/main/fold_1/confusion_matrix_absolute.png` |
| **Table IX / X** | `bash scripts/run_control_model.sh` | `results/control/` |
| **Figure 3** | (produced by Table IX) | `results/control/fold_1/confusion_matrix_absolute.png` |
| **Table XI** | `python -m src.utils.compare_models` | `results/comparison/table_xi.{csv,md}` |

## 📊 Key Results

| Model | Overall Acc. | Macro F1 | AUC-ROC | Sensitivity (S) | Sensitivity (F) |
|---|---|---|---|---|---|
| **Main (balanced)** | **99.70%** | **99.70%** | **99.99%** | **99.98%** | **100.00%** |
| Control (no balancing) | 98.29% | 93.75% | 99.56% | 71.80% | 90.80% |

Controlled experiment demonstrates **+5.95 pts macro F1** and **+28.18 pts
sensitivity on supraventricular beats** attributable to balancing + augmentation.

- 5-fold stratified cross-validation (σ < 0.00025 in macro F1)
- MIT-BIH Arrhythmia Database, 5 classes (N, S, V, F, Q)
- Custom 3-layer 2D CNN on 128×128 CWT spectrograms

![Confusion Matrix](reports/figures/figure2_confusion_main.png)

---

## 🧠 Method Overview

1. **Preprocessing** — R-peak-centered 0.6 s segmentation (216 samples @ 360 Hz) + 6th-order polynomial baseline removal.
2. **Representation** — Continuous Wavelet Transform (complex Morlet, ω₀ = 6 rad/s) → 128×128 time-frequency spectrograms.
3. **Balancing** — Brightness/contrast/horizontal-flip augmentation applied only to minority classes (N=150,680 per class; 753,400 total beats).
4. **Modeling** — Custom CNN (64→128→256 filters, 3×3 kernels, BatchNorm, MaxPool, Dropout 0.15/0.3, 256-unit FC) trained with AdamW, mixed precision, and early stopping.
5. **Evaluation** — Stratified 5-fold CV; macro F1, per-class AUC-ROC, and Average Precision.

---

## 🔁 Reproducing the paper

```bash
# 1. Data pipeline
bash scripts/run_preprocessing.sh                # → data/processed/ECG_spectrograms/
bash scripts/run_balance.sh                      # → 753,400 balanced images

# 2. Exploratory (Table III)
python src/models/mobilenet_fusion.py --config configs/mobilenetv2.yaml

# 3. Hyperparameter search (Table VI hyperparams)
python src/models/search_hyperparameters.py --trials 6 --use-wandb
python src/utils/extract_best_params.py          # → configs/custom_cnn.yaml

# 4. Control model (Tables IX–X, Figure 3)
python src/models/train_crossval.py --config configs/control_model.yaml

# 5. Main model (Tables VI–VIII, Figure 2)
python src/models/train_crossval.py --config configs/main_model.yaml

# 6. Aggregate + compare (Table XI)
python src/utils/summarize_results.py --results-dir results/main
python src/utils/summarize_results.py --results-dir results/control
python src/utils/compare_models.py
