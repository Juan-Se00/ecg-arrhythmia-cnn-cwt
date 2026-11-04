"""Verify ECGDataset handles augmentation flag correctly."""
from pathlib import Path
import pytest
from src.data.dataset import ECGDataset


def _make_fake_spectrogram(root: Path, derivation, cls, name, is_aug=False):
    d = root / derivation / cls
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}{'_aug_1' if is_aug else ''}.png").write_bytes(b'\x89PNG\r\n')


def test_filters_augmented_images(tmp_path):
    _make_fake_spectrogram(tmp_path, 'MLII', 'Normal', 'r1')
    _make_fake_spectrogram(tmp_path, 'MLII', 'Normal', 'r2', is_aug=True)

    with_aug = ECGDataset(tmp_path, ['Normal', 'Fusion'], include_augmented=True)
    without_aug = ECGDataset(tmp_path, ['Normal', 'Fusion'], include_augmented=False)

    assert len(with_aug) == 2
    assert len(without_aug) == 1
