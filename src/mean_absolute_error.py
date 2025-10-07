import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    This loss function calculates the average absolute difference between true and predicted values.
    It provides an intuitive measure of prediction error that treats all deviations equally and is
    less sensitive to outliers compared to squared error metrics.

    Attributes:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Output:
        float: The mean absolute error across all samples.
    """
    return float(np.mean(np.abs(y_true - y_pred)))
    # Once again, vectorized operations w/ NumPy arrays.
