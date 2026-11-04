"""6th-order polynomial baseline removal (paper Eq. 2-3)."""
import numpy as np

def remove_baseline_wander(ecg_signal: np.ndarray, degree: int = 6) -> np.ndarray:
    x = np.arange(len(ecg_signal))
    coeffs = np.polyfit(x, ecg_signal, degree)
    return ecg_signal - np.polyval(coeffs, x)
