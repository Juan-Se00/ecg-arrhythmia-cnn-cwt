"""Metrics used across evaluation scripts (paper §III.E.9, Eq. 20-23)."""
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)


@torch.inference_mode()
def predict_all(model, dataset, device, batch_size=64):
    """Run inference → (y_true, y_probs, y_pred)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval().to(device)
    y_true, y_probs, y_pred = [], [], []
    for x, y in loader:
        x = x.to(device)
        logits = torch.clamp(model(x), -50, 50)  # prevent overflow
        probs = F.softmax(logits, dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(probs.argmax(1).cpu().numpy())
        y_probs.append(probs.cpu().numpy())
    return np.array(y_true), np.vstack(y_probs), np.array(y_pred)


def compute_metrics(model, dataset, class_names, device) -> dict:
    y_true, y_probs, y_pred = predict_all(model, dataset, device)
    n = len(class_names)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)

    bin_labels = np.eye(n)[y_true]
    roc_auc, pr_ap = {}, {}
    for i in range(n):
        fpr, tpr, _ = roc_curve(bin_labels[:, i], y_probs[:, i])
        roc_auc[i] = float(auc(fpr, tpr))
        _, _, _ = precision_recall_curve(bin_labels[:, i], y_probs[:, i])
        pr_ap[i] = float(average_precision_score(bin_labels[:, i], y_probs[:, i]))

    return {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'f1_scores': f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        'roc_auc': roc_auc,
        'pr_ap': pr_ap,
        'accuracy': float(report['accuracy']),
        'macro_f1': float(report['macro avg']['f1-score']),
        'weighted_f1': float(report['weighted avg']['f1-score']),
    }


def save_metrics_json(metrics: dict, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'classification_report.json').write_text(
        json.dumps(metrics['classification_report'], indent=2)
    )
    (out_dir / 'metrics_summary.json').write_text(
        json.dumps({k: metrics[k] for k in
                    ('accuracy', 'macro_f1', 'weighted_f1', 'roc_auc', 'pr_ap')},
                   indent=2)
    )
