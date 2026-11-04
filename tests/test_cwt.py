"""Verify CWT output shape and range (paper Section III.B.3, Eq. 4)."""
import numpy as np
from src.features.generate_cwt import compute_spectrogram


def test_spectrogram_shape(ecg_1d_signal):
    seg = ecg_1d_signal[:216]
    freqs, power_db = compute_spectrogram(seg, fs=360)
    assert power_db.ndim == 2
    assert power_db.shape[0] == 127          # np.arange(1, 128)
    assert power_db.shape[1] == len(seg)
    assert freqs.shape == (127,)


def test_no_nan_inf(ecg_1d_signal):
    seg = ecg_1d_signal[:216]
    _, power_db = compute_spectrogram(seg, fs=360)
    assert np.isfinite(power_db).all()
