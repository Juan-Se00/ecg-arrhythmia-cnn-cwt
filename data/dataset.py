"""Shared PyTorch Dataset (replaces 4 copies in the original code)."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


class ECGDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        classes: list[str],
        img_size: tuple[int, int] = (128, 128),
        derivation: str | None = None,
        include_augmented: bool = True,
        indices: list[int] | None = None,
    ):
        self.img_size = img_size
        self.file_paths: list[Path] = []
        self.labels: list[str] = []

        derivations = [derivation] if derivation else None
        roots = (Path(data_root) / derivation) if derivation else None

        if derivation:
            roots_iter = [(Path(data_root) / derivation, cls) for cls in classes]
        else:
            roots_iter = [
                (Path(data_root) / d / cls, cls)
                for d in (Path(data_root).iterdir())
                if d.is_dir()
                for cls in classes
            ]

        for folder, cls in roots_iter:
            if not folder.exists():
                continue
            for f in folder.iterdir():
                if not f.suffix.lower() in ('.png', '.jpg'):
                    continue
                if not include_augmented and '_aug_' in f.name:
                    continue
                self.file_paths.append(f)
                self.labels.append(cls)

        if not self.file_paths:
            raise RuntimeError(f"No images found under {data_root}")

        self.label_encoder = LabelEncoder().fit(classes)
        self.encoded_labels = self.label_encoder.transform(self.labels)

        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.encoded_labels = self.encoded_labels[indices]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.file_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(self.img_size, dtype=np.uint8)
        img = cv2.resize(img, self.img_size)
        x = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        y = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return x, y
