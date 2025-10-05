import numpy as np


def jaccard_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Jaccard Loss."""
    a = y_true.astype(bool)
    b = y_pred.astype(bool)
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(1 - intersection / union)
