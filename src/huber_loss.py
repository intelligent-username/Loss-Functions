import numpy as np


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float) -> float:
    """
    Huber Loss.

    This loss function combines the advantages of mean squared error and mean absolute error.
    It behaves quadratically for small errors and linearly for large errors, reducing the influence
    of outliers while remaining smooth and differentiable.

    Attributes:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        delta (float): Threshold parameter that determines the transition between quadratic and linear behavior.

    Output:
        float: The mean Huber loss across all samples.
    """
    if delta <= 0:
        raise ValueError("delta must be positive.")
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for huber_loss.")
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    quadratic = 0.5 * diff**2
    linear = delta * abs_diff - 0.5 * delta**2
    return float(np.mean(np.where(abs_diff <= delta, quadratic, linear)))
