import numpy as np
from typing import Union


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    This loss function calculates the average of the squared differences between true and predicted
    values. It penalizes larger errors more heavily due to the squaring, making it sensitive to
    outliers, and is widely used in regression tasks.

    Attributes:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Output:
        float: The mean squared error across all samples.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for mean_squared_error.")
    diff = y_true - y_pred
    # Note: since these are arrays, we're doing vectorized subtraction here.
    return float(np.mean(diff * diff))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for mean_absolute_error.")
    return float(np.mean(np.abs(y_true - y_pred)))
