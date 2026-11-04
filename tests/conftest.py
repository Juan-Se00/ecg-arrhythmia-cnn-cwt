"""Shared pytest fixtures."""
import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def dummy_spectrogram():
    """Random 128×128 grayscale tensor in [0, 1]."""
    return torch.rand(1, 1, 128, 128)


@pytest.fixture
def ecg_1d_signal():
    """Synthetic 1-D ECG-like signal, 10 s @ 360 Hz."""
    import numpy as np
    fs = 360
    t = np.arange(10 * fs) / fs
    return np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(len(t))
