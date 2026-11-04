"""Random hyperparameter search (paper §III.E.5, ~6 trials)."""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.dataset import ECGDataset
from ..models.custom_cnn import ECGCNN
from ..models.trainer import ModelTrainer
from ..utils.checkpoint import save_checkpoint
from ..utils.config import load_config
from ..utils.seeding import set_seeds


SEARCH_SPACE = {
    'model': {
        'dropout':     [0.3, 0.4],
        'filters':     [32, 64],
        'conv_layers': [3, 4],
        'dense_units': [256, 512],
    },
    'training': {
        'batch_size':    [32, 64],
        'learning_rate': [1e-4, 3e-4],
        'weight_decay':  [1e-4, 3e-4],
    },
}


def sample_params(rng: random.Random) -> dict:
    m = {k: rng.choice(v) for k, v in SEARCH_SPACE['model'].items()}
    t = {k: rng.choice(v) for k, v in SEARCH_SPACE['training'].items()}
    return {**m, **t}


def run_trial(params: dict, train_ds, val_ds, cfg, trial_id: int, use_wandb: bool):
    if use_wandb:
        import wandb
        wandb.init(project=cfg.get('wandb_project', 'ecg-search'), config=params)

    model = ECGCNN(
        dropout=params['dropout'],
        filters=params['filters'],
        conv_layers=params['conv_layers'],
        dense_units=params['dense_units'],
        img_size=tuple(cfg['img_size']),
        num_classes=cfg['num_classes'],
    )
    trainer = ModelTrainer(model, train_ds, val_ds, params, cfg['device'])
    _, best_loss = trainer.train(epochs=cfg.get('max_epochs', 30))

    if use_wandb:
        wandb.log({'best_val_loss': best_loss})
        wandb.finish()

    return best_loss


def main(args):
    cfg = load_config(args.config)
    set_seeds(cfg['seed'])
    rng = random.Random(cfg['seed'])

    data = ECGDataset(cfg['paths']['data_root'], cfg['classes'],
                      tuple(cfg['img_size']), include_augmented=True)

    from sklearn.model_selection import train_test_split
    train_idx, val_idx = train_test_split(
        list(range(len(data))), test_size=0.2, stratify=data.encoded_labels,
        random_state=cfg['seed'],
    )
    from torch.utils.data import Subset
    train_ds, val_ds = Subset(data, train_idx), Subset(data, val_idx)

    best_loss, best_params = float('inf'), None
    for trial in range(1, args.trials + 1):
        params = sample_params(rng)
        print(f"\n=== Trial {trial}/{args.trials} ===")
        print(json.dumps(params, indent=2))
        loss = run_trial(params, train_ds, val_ds, cfg, trial, args.use_wandb)

        if loss < best_loss:
            best_loss, best_params = loss, params
            save_checkpoint(
                Path(cfg['paths']['models_root']) / 'best_overall_loss_model.pth',
                epoch=0,
                model_state=None,
                optimizer_state=None,
                scaler_state=None,
                scheduler_state=None,
                best_val_loss=best_loss,
                params=best_params,
            )

    print(f"\nBest val loss: {best_loss:.4f}")
    print(json.dumps(best_params, indent=2))
    with open(Path(cfg['paths']['models_root']) / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--trials', type=int, default=6)
    p.add_argument('--use-wandb', action='store_true')
    main(p.parse_args())
