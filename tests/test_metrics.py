"""Sanity checks for metric computation."""
import numpy as np
from sklearn.metrics import f1_score


def test_perfect_predictions_give_macro_f1_one():
    y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    y_pred = y_true.copy()
    assert f1_score(y_true, y_pred, average='macro') == 1.0


def test_confusion_matrix_shape():
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix([0, 1, 2, 0], [0, 1, 1, 0], labels=list(range(5)))
    assert cm.shape == (5, 5)
