# src/models/train_crossval.py
"""5-fold stratified cross-validation (paper §III.E.6, §IV.A, §IV.C).

Used for BOTH the main (balanced) and control (unbalanced) models:
  python src/models/train_crossval.py --config configs/main_model.yaml
  python src/models/train_crossval.py --config configs/control_model.yaml
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
from ..models.custom_cnn import ECGCNN
from ..models.trainer import ModelTrainer
from ..utils.config import load_config
from ..utils.metrics import compute_metrics, save_metrics_json
from ..utils.plots import plot_confusion_matrix
from ..utils.seeding import set_seeds


def run_fold(cfg, dataset, train_idx, test_idx, fold, fold_dir):
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25,
        stratify=dataset.encoded_labels[train_idx],
        random_state=cfg['seed'],
    )
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    print(f"\nFold {fold}: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}")

    model = ECGCNN(**{k: cfg['model'][k] for k in ('dropout', 'filters', 'conv_layers', 'dense_units')},
                   img_size=tuple(cfg['img_size']), num_classes=cfg['num_classes'])
    trainer = ModelTrainer(model, train_ds, val_ds, cfg['training'], cfg['device'])
    best_state, best_val_loss = trainer.train(
        epochs=cfg['training'].get('epochs', 30),
        patience=cfg['training'].get('patience', 5),
    )

    # Evaluate
    model.load_state_dict(best_state)
    metrics = compute_metrics(model, test_ds, cfg['classes'], cfg['device'])

    # Persist
    save_metrics_json(metrics, fold_dir)
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        cfg['classes'],
        fold_dir / 'confusion_matrix_absolute.png',
        normalize=False,
    )
    plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        cfg['classes'],
        fold_dir / 'confusion_matrix_normalized.png',
        normalize=True,
    )
    torch.save(best_state, fold_dir / f'model_fold_{fold}.pth')
    return metrics, best_val_loss


def main(args):
    cfg = load_config(args.config)
    set_seeds(cfg['seed'])

    results_dir = Path(cfg['paths']['results_root']) / args.experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results → {results_dir}")
    print(f"Balanced: {cfg['training']['include_augmented']}")

    dataset = ECGDataset(
        data_root=cfg['paths']['data_root'],
        classes=cfg['classes'],
        img_size=tuple(cfg['img_size']),
        include_augmented=cfg['training']['include_augmented'],
    )
    print(f"Dataset: {len(dataset)} images")

    skf = StratifiedKFold(
        n_splits=cfg['training']['k_folds'],
        shuffle=True,
        random_state=cfg['seed'],
    )

    all_metrics, summary_rows = [], []
    for fold, (train_idx, test_idx) in enumerate(
        skf.split(dataset.file_paths, dataset.encoded_labels), start=1
    ):
        fold_dir = results_dir / f'fold_{fold}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        metrics, val_loss = run_fold(cfg, dataset, train_idx, test_idx, fold, fold_dir)
        all_metrics.append(metrics)
        summary_rows.append({
            'fold': fold,
            'val_loss': val_loss,
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1'],
        })
        print(f"Fold {fold} — Acc {metrics['accuracy']:.4f} | Macro F1 {metrics['macro_f1']:.4f}")

    # Consolidate
    summary = pd.DataFrame(summary_rows)
    summary.loc['mean'] = summary.mean(numeric_only=True)
    summary.loc['std'] = summary.std(numeric_only=True)
    summary.to_csv(results_dir / 'final_results' / 'cross_validation_summary.csv')

    consolidated = {
        'overall_accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
        'overall_macro_f1': float(np.mean([m['macro_f1'] for m in all_metrics])),
        'roc_auc': {i: float(np.mean([m['roc_auc'][i] for m in all_metrics]))
                    for i in range(cfg['num_classes'])},
        'pr_ap':   {i: float(np.mean([m['pr_ap'][i] for m in all_metrics]))
                    for i in range(cfg['num_classes'])},
    }
    (results_dir / 'final_results').mkdir(exist_ok=True)
    (results_dir / 'final_results' / 'consolidated_metrics.json').write_text(
        json.dumps(consolidated, indent=2)
    )

    print(f"\n{'='*50}")
    print(f"Mean accuracy:  {consolidated['overall_accuracy']:.4f}")
    print(f"Mean macro F1:  {consolidated['overall_macro_f1']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--experiment-name', required=True,
                   help='Subfolder under results/ (e.g. "main" or "control")')
    main(p.parse_args())
