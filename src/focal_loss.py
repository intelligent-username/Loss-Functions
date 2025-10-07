import numpy as np


def focal_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> float:
    """
    Focal Loss.

    This loss function modifies cross-entropy to address class imbalance by focusing on hard-to-classify
    examples. It down-weights easy examples and emphasizes difficult ones using modulating factors,
    helping improve performance on minority classes or challenging predictions.

    Attributes:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_pred (np.ndarray): Predicted probabilities for the positive class.
        alpha (float): Weighting factor for positive examples, default 0.25.
        gamma (float): Focusing parameter that reduces loss for easy examples, default 2.0.

    Output:
        float: The average focal loss across all samples.
    """

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Standard binary cross-entropy components
    ce_pos = -y_true * np.log(y_pred)
    ce_neg = -(1 - y_true) * np.log(1 - y_pred)

    # Apply focal modulation
    focal_pos = (1 - y_pred) ** gamma * ce_pos
    focal_neg = (y_pred) ** gamma * ce_neg

    loss = alpha * focal_pos + (1 - alpha) * focal_neg
    return float(np.mean(loss))
