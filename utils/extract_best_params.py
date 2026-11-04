# src/utils/extract_best_params.py
"""Extract best hyperparameters from a search checkpoint (paper §III.E.5)."""
import argparse
import json
from pathlib import Path
import torch

REQUIRED_KEYS = ['dropout', 'filters', 'conv_layers', 'dense_units']


def _filter(p: dict) -> dict:
    return {k: p[k] for k in REQUIRED_KEYS}


def from_checkpoint(checkpoint_path: Path) -> dict:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'params' not in ckpt:
        raise KeyError(f"No 'params' key in {checkpoint_path}. "
                       "Use --estimate to infer from weights.")
    return _filter(ckpt['params'])


def estimate_from_weights(model_path: Path) -> dict:
    """Fallback: infer architecture from a saved state_dict."""
    sd = torch.load(model_path, map_location='cpu')
    conv_keys = sorted(
        [k for k in sd if 'conv_layers' in k and k.endswith('.0.weight')],
        key=lambda k: int(k.split('.')[1]),
    )
    first_conv = sd[conv_keys[0]]
    fc_key = next(k for k in sd if k == 'fc.0.weight')
    return {
        'dropout': 0.3,  # not recoverable from state_dict
        'filters': first_conv.shape[0],
        'conv_layers': len(conv_keys),
        'dense_units': sd[fc_key].shape[0],
    }


def main(args):
    out = Path(args.output)
    if args.checkpoint:
        params = from_checkpoint(Path(args.checkpoint))
        print(f"[from checkpoint] {json.dumps(params, indent=2)}")
    else:
        params = estimate_from_weights(Path(args.weights))
        print(f"[estimated from weights] {json.dumps(params, indent=2)}")
    out.write_text(json.dumps(params, indent=2))
    print(f"Saved → {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint', help='search checkpoint with "params" key')
    g.add_argument('--weights', help='.pth state_dict to estimate from')
    p.add_argument('--output', default='best_params.json')
    main(p.parse_args())
