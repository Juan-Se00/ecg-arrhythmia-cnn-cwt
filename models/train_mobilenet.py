"""Train/evaluate MobileNet experiments (paper Table III)."""
import argparse
from pathlib import Path

from ..data.dataset import ECGDataset
from ..models.mobilenet import MobileNetDerivation, FusionModel
from ..models.trainer import ModelTrainer
from ..utils.config import load_config
from ..utils.metrics import compute_metrics
from ..utils.seeding import set_seeds


def build_model(experiment: str, cfg: dict):
    if experiment == 'individual':
        return MobileNetDerivation(cfg['model']['dropout'], cfg['num_classes'])
    if experiment == 'fusion':
        return FusionModel(cfg['model']['dropout'], cfg['num_classes'])
    raise ValueError(experiment)


def main(args):
    cfg = load_config(args.config)
    set_seeds(cfg['seed'])

    if args.experiment == 'individual':
        for derivation in cfg['derivations']:
            run_experiment(cfg, derivation, args.experiment)
    else:
        run_experiment(cfg, None, 'fusion')


def run_experiment(cfg, derivation, experiment):
    # ... uses ModelTrainer + compute_metrics
    # ... same 5-fold loop as train_crossval.py
    pass
