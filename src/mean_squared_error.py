import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    diff = y_true - y_pred
    return float(np.mean(diff * diff))
