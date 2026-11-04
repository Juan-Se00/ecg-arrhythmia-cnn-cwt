"""Train/evaluate MobileNet experiments (paper Table III).

Usage:
  python -m src.models.train_mobilenet --config configs/mobilenetv2.yaml --experiment fusion
  python -m src.models.train_mobilenet --config configs/mobilenetv2.yaml --experiment individual
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Subset

from ..data.dataset import ECGDataset
from ..models.mobilenet import MobileNetDerivation, FusionModel
from ..models.trainer import ModelTrainer
from ..utils.config import load_config
from ..utils.metrics import compute_metrics, save_metrics_json
from ..utils.plots import plot_confusion_matrix
from ..utils.seeding import set_seeds


def build_model(experiment: str, cfg: dict):
    if experiment == 'individual':
        return MobileNetDerivation(cfg['model']['dropout'], cfg['num_classes'])
    if experiment == 'fusion':
        return FusionModel(cfg['model']['dropout'], cfg['num_classes'])
    raise ValueError(f"Unknown experiment: {experiment}")


def run_fold(cfg, derivation, experiment, train_idx, test_idx, fold, fold_dir):
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25,
        stratify=cfg['_encoded_labels'][train_idx],
        random_state=cfg['seed'],
    )
    train_ds = Subset(cfg['_dataset'], train_idx)
    val_ds = Subset(cfg['_dataset'], val_idx)
    test_ds = Subset(cfg['_dataset'], test_idx)

    model = build_model(experiment, cfg)
    trainer = ModelTrainer(model, train_ds, val_ds, cfg['training'], cfg['device'])
    best_state, _ = trainer.train(
        epochs=cfg['training'].get('epochs', 30),
        patience=cfg['training'].get('patience', 5),
    )

    model.load_state_dict(best_state)
    metrics = compute_metrics(model, test_ds, cfg['classes'], cfg['device'])
    save_metrics_json(metrics, fold_dir)
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        cfg['classes'],
        fold_dir / 'confusion_matrix.png',
    )
    return metrics


def run_experiment(cfg, derivation, experiment, results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = ECGDataset(
        data_root=cfg['paths']['data_root'],
        classes=cfg['classes'],
        img_size=tuple(cfg['img_size']),
        derivation=derivation,
        include_augmented=False,
    )
    cfg['_dataset'] = dataset
    cfg['_encoded_labels'] = dataset.encoded_labels

    skf = StratifiedKFold(
        n_splits=cfg['training']['k_folds'],
        shuffle=True, random_state=cfg['seed'],
    )

    all_metrics = []
    for fold, (tr, te) in enumerate(
        skf.split(dataset.file_paths, dataset.encoded_labels), 1
    ):
        fold_dir = results_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_fold(cfg, derivation, experiment, tr, te, fold, fold_dir)
        all_metrics.append(metrics)
        print(f"Fold {fold}: Acc={metrics['accuracy']:.4f} | F1={metrics['macro_f1']:.4f}")

    consolidated = {
        'overall_accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
        'overall_macro_f1': float(np.mean([m['macro_f1'] for m in all_metrics])),
        'roc_auc': {
            i: float(np.mean([m['roc_auc'][i] for m in all_metrics]))
            for i in range(cfg['num_classes'])
        },
        'pr_ap': {
            i: float(np.mean([m['pr_ap'][i] for m in all_metrics]))
            for i in range(cfg['num_classes'])
        },
    }
    (results_dir / 'final_results').mkdir(exist_ok=True)
    (results_dir / 'final_results' / 'consolidated_metrics.json').write_text(
        json.dumps(consolidated, indent=2)
    )
    print(f"\n{experiment}/{derivation or 'fusion'}: "
          f"Acc={consolidated['overall_accuracy']:.4f} | "
          f"F1={consolidated['overall_macro_f1']:.4f}")


def main(args):
    cfg = load_config(args.config)
    set_seeds(cfg['seed'])

    results_root = Path(cfg['paths']['results_root']) / 'mobilenet' / args.experiment

    if args.experiment == 'individual':
        for derivation in cfg['derivations']:
            print(f"\n=== {derivation} ===")
            run_experiment(cfg, derivation, 'individual', results_root / derivation)
    else:  # fusion
        print("\n=== Fusion (MLII + V5) ===")
        run_experiment(cfg, None, 'fusion', results_root)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--experiment', required=True, choices=['individual', 'fusion'])
    main(p.parse_args())
