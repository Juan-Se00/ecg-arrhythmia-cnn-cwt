"""Sanity checks for 0.6 s R-peak-centered segmentation (paper Eq. 1)."""
import numpy as np
from src.data.segment_beats import segment_beat


def test_window_length():
    sig = np.arange(10000, dtype=float)
    seg = segment_beat(sig, r_peak=5000, fs=360)
    assert seg is not None
    assert len(seg) == int(0.6 * 360) == 216


def test_asymmetric_window():
    sig = np.arange(10000, dtype=float)
    seg = segment_beat(sig, r_peak=5000, fs=360)
    # 0.2 s before → 72 samples; 0.4 s after → 144 samples
    assert seg[0] == 5000 - 72
    assert seg[-1] == 5000 + 144 - 1


def test_out_of_bounds_returns_none():
    sig = np.arange(100, dtype=float)
    assert segment_beat(sig, r_peak=5, fs=360) is None
    assert segment_beat(sig, r_peak=99, fs=360) is None
