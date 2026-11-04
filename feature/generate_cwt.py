"""Continuous Wavelet Transform → 128×128 spectrograms (paper Eq. 4)."""
import numpy as np
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

def compute_spectrogram(beat, fs=360, wavelet='cmor'):
    scales = np.arange(1, 128)
    coeffs, freqs = pywt.cwt(beat, scales, wavelet, sampling_period=1/fs)
    power_db = 10 * np.log10(np.abs(coeffs) ** 2 + 1e-10)
    return freqs, power_db

def save_spectrogram_png(power_db, freqs, fs, length, out_path: Path, title=''):
    plt.figure(figsize=(4, 4), dpi=100)
    plt.imshow(power_db,
               extent=[0, length / fs, freqs.min(), freqs.max()],
               aspect='auto', cmap='jet')
    plt.xlabel('Tiempo [s]'); plt.ylabel('Frecuencia [Hz]')
    plt.title(title); plt.colorbar(label='Potencia (dB)')
    plt.tight_layout(pad=0)
    tmp = out_path.parent / '_tmp.png'
    plt.savefig(tmp, bbox_inches='tight', pad_inches=0); plt.close()
    Image.open(tmp).resize((128, 128), Image.LANCZOS).save(out_path)
    tmp.unlink()
