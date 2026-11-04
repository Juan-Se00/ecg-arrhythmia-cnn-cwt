# Multiclass ECG Arrhythmia Classification via CWT Spectro-Temporal Representations and 2D CNNs

Reference implementation for the paper:

> **Ardila Parra, J.S.** & Orjuela Cañón, A.D. (2024).
> *Multiclass Classification of Cardiac Arrhythmias Using Two-Dimensional
> Convolutional Neural Networks and ECG Spectro-Temporal Representations.*
> Undergraduate thesis, Universidad del Rosario & Escuela Colombiana de
> Ingeniería Julio Garavito.
> [📄 Full paper](reports/paper/Ardila_Parra_2024_ECG_Arrhythmia_Classification.pdf)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

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

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/<your-user>/ecg-arrhythmia-cnn-cwt.git
cd ecg-arrhythmia-cnn-cwt

# 2. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Data (see data/README.md)
# Download MIT-BIH to data/raw/ then:
python src/data/load_mitbih.py
python src/data/remove_baseline.py
python src/data/segment_beats.py
python src/features/generate_cwt.py

# 4. Balance
python src/balancing/augment_and_balance.py

# 5. Train main model (balanced)
python src/models/train.py --config configs/custom_cnn.yaml

# 6. Train control model (no balancing)
python src/models/train.py --config configs/custom_cnn.yaml --no-balance

# 7. Evaluate
python src/models/evaluate.py --checkpoint models/best_fold1.pth
