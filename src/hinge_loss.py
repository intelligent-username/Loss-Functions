import numpy as np


def hinge_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Hinge Loss.

    This loss function is commonly used in support vector machines to maximize the margin between
    classes. It penalizes misclassifications and predictions that are too close to the decision
    boundary, encouraging models to create clear separation between positive and negative classes.

    Attributes:
        y_true (np.ndarray): True labels, typically -1 or 1.
        y_pred (np.ndarray): Predicted values, usually raw scores before thresholding.

    Output:
        float: The average hinge loss across all samples.
    """

    losses = np.maximum(0, 1 - y_true * y_pred)
    return np.mean(losses)
