# src/balancing/augment_and_balance.py
"""Controlled augmentation + balancing (paper §III.D, Table IV)."""
import argparse
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm

ORIGINAL_COUNTS = {
    'Normal': 75340,
    'Fusion': 1784,
    'Supraventricular': 2714,
    'Unknown': 18247,
    'Ventricular': 14490,
}


def augment_image(image: Image.Image) -> Image.Image:
    """Brightness / contrast / horizontal flip (paper Eq. 7-9)."""
    ops = [
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.9, 1.1)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.9, 1.1)),
        ImageOps.mirror,
    ]
    return random.choice(ops)(image)


def delete_previous_augmentations(data_root: Path):
    for lead_dir in data_root.iterdir():
        if not lead_dir.is_dir():
            continue
        for cls_dir in lead_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.glob('*_aug_*.png'):
                f.unlink()


def balance_class(class_dir: Path, target: int, class_name: str) -> int:
    originals = [f for f in class_dir.iterdir() if '_aug_' not in f.name]
    if not originals:
        return 0
    needed = target - len(originals)
    if needed <= 0:
        return 0

    for i in tqdm(range(needed), desc=f"  {class_name}"):
        src = random.choice(originals)
        img = augment_image(Image.open(src).convert('L'))
        img.save(class_dir / f"{src.stem}_aug_{i+1}.png")
    return needed


def main(args):
    random.seed(42)
    data_root = Path(args.data_root)
    target = args.target

    print(f"Removing previous augmentations under {data_root}...")
    delete_previous_augmentations(data_root)

    for lead_dir in sorted(data_root.iterdir()):
        if not lead_dir.is_dir():
            continue
        print(f"\n[{lead_dir.name}]")
        for cls_dir in sorted(lead_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            n = balance_class(cls_dir, target, cls_dir.name)
            print(f"  {cls_dir.name}: +{n} augmented")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', required=True)
    p.add_argument('--target', type=int, default=75340)
    main(p.parse_args())
