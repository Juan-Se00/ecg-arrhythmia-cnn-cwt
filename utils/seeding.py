"""Deterministic seeding for torch/numpy/random (paper Section III.F.2)."""
import os
import random
import numpy as np
import torch


def set_seeds(seed: int = 42, deterministic: bool = True) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # torch 2.x: fast but stable matmul
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass
