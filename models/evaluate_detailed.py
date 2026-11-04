"""Full evaluation: ROC, PR curves, per-class metrics."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from ..data.dataset import ECGDataset
from ..models import ARCH_REGISTRY
from ..utils.config import load_config
from ..utils.metrics import compute_metrics, save_metrics_json
from ..utils.plots import plot_confusion_matrix


def main(args):
    cfg = load_config(args.config)
    params = json.loads(Path(args.params).read_text())
    model_cls = ARCH_REGISTRY[args.arch]
    model = model_cls(**params, img_size=tuple(cfg['img_size']),
                      num_classes=cfg['num_classes'])
    model.load_state_dict(torch.load(args.model_path, map_location=cfg['device']))

    dataset = ECGDataset(cfg['paths']['data_root'], cfg['classes'],
                         tuple(cfg['img_size']), include_augmented=False)

    out_dir = Path(args.output)
    metrics = compute_metrics(model, dataset, cfg['classes'], cfg['device'])
    save_metrics_json(metrics, out_dir)
    plot_confusion_matrix(np.array(metrics['confusion_matrix']),
                          cfg['classes'], out_dir / 'confusion_matrix.png')

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"Results → {out_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--arch', required=True, choices=list(ARCH_REGISTRY))
    p.add_argument('--params', required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--output', default='evaluation_results')
    main(p.parse_args())
