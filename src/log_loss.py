import numpy as np


def log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Log Loss.

    This loss function, also known as binary cross-entropy, evaluates how well predicted probabilities
    match true binary labels. It penalizes confident incorrect predictions more heavily than uncertain
    ones, promoting well-calibrated probability estimates.

    Attributes:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities for the positive class.

    Output:
        float: The average log loss across all samples.
    """
    n = len(y_true)
    epsilon = 1e-15  # To avoid log(0)

    # NumPy's 'clipping': avoids inputting a 0 into log, and avoid 100% 'confidence'
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / n
