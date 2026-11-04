"""Shared plotting helpers."""
import itertools
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, class_names, out_path: Path, normalize=False):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt, title = '.2f', 'Confusion Matrix (Normalized)'
    else:
        fmt, title = 'd', 'Confusion Matrix (Absolute)'

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title); fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticklabels(class_names)
    ax.set_ylabel('True Label'); ax.set_xlabel('Predicted Label')

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), ha='center', fontsize=10,
                color='white' if cm[i, j] > thresh else 'black')

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
