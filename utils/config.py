"""Single source of truth for configuration (replaces 11 copies of Config)."""
from dataclasses import dataclass, field
from pathlib import Path
import yaml
import torch

CLASSES = ['Fusion', 'Normal', 'Supraventricular', 'Unknown', 'Ventricular']
DERIVATIONS = ['MLII', 'V5']
NUM_CLASSES = len(CLASSES)


def _resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str | Path) -> dict:
    """Load YAML config, resolving `inherits:` recursively."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if "inherits" in cfg:
        parent_path = Path(path).parent / cfg.pop("inherits")
        parent = load_config(parent_path)
        parent.update(cfg)
        cfg = parent

    cfg.setdefault("seed", 42)
    cfg.setdefault("classes", CLASSES)
    cfg.setdefault("num_classes", NUM_CLASSES)
    cfg.setdefault("derivations", DERIVATIONS)
    cfg["device"] = _resolve_device(cfg.get("device", "auto"))
    return cfg
