"""Aggregate 5-fold results into a consolidated report."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd


def load_fold(fold_dir: Path) -> dict | None:
    summary = fold_dir / 'metrics_summary.json'
    report = fold_dir / 'classification_report.json'
    if not summary.exists():
        return None
    m = json.loads(summary.read_text())
    if 'roc_auc' in m:
        m['roc_auc'] = {int(k): v for k, v in m['roc_auc'].items()}
    if 'pr_ap' in m:
        m['pr_ap'] = {int(k): v for k, v in m['pr_ap'].items()}
    m['classification_report'] = json.loads(report.read_text()) if report.exists() else {}
    return m


def main(args):
    results_dir = Path(args.results_dir)
    folds = [load_fold(results_dir / f'fold_{i}') for i in range(1, args.k_folds + 1)]
    folds = [f for f in folds if f]

    if not folds:
        raise SystemExit(f"No folds found under {results_dir}")

    best = max(folds, key=lambda x: x['macro_f1'])
    avg_acc = np.mean([f['accuracy'] for f in folds])
    avg_f1 = np.mean([f['macro_f1'] for f in folds])

    out = results_dir / 'consolidated_report'
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([
        {'Fold': i, 'Accuracy': f['accuracy'], 'Macro_F1': f['macro_f1'],
         'Weighted_F1': f['weighted_f1']}
        for i, f in enumerate(folds, 1)
    ])
    df.loc['Mean'] = df.mean(numeric_only=True)
    df.loc['Std'] = df.std(numeric_only=True)
    df.to_csv(out / 'cross_validation_summary.csv', index=False)

    report = {
        'best_fold': int(best is folds[0]),
        'average_accuracy': float(avg_acc),
        'average_macro_f1': float(avg_f1),
        'per_class_roc_auc': {
            k: float(np.mean([f['roc_auc'][k] for f in folds]))
            for k in folds[0]['roc_auc']
        },
        'per_class_pr_ap': {
            k: float(np.mean([f['pr_ap'][k] for f in folds]))
            for k in folds[0]['pr_ap']
        },
    }
    (out / 'consolidated_summary.json').write_text(json.dumps(report, indent=2))
    print(df)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results-dir', required=True)
    p.add_argument('--k-folds', type=int, default=5)
    main(p.parse_args())
