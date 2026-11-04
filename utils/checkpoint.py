"""Save/load training checkpoints uniformly (fixes fragile checkpoint dicts)."""
import torch
from pathlib import Path

FIELDS = ['epoch', 'model_state', 'optimizer_state', 'scaler_state',
          'scheduler_state', 'best_val_loss', 'params']


def save_checkpoint(path, **state):
    missing = set(FIELDS) - set(state)
    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")
    torch.save(state, path)


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device)
