import numpy as np

def jaccard_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-15) -> float:
    """
    Jaccard Loss.

    This loss function, also known as Intersection over Union (IoU) loss, measures the overlap
    between predicted and true sets. It calculates 1 minus the Jaccard index, emphasizing
    accurate region matching in tasks like image segmentation.

    Attributes:
        y_true (np.ndarray): True binary values (0 or 1).
        y_pred (np.ndarray): Predicted values, can be binary or soft probabilities.
        eps (float): Small epsilon value to avoid division by zero, default 1e-15.

    Output:
        float: The Jaccard loss, where 0 indicates perfect overlap and higher values indicate poorer overlap.
    """
    y_true = y_true.astype(float)
    y_pred = y_pred.astype(float)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return 1.0 - (intersection + eps) / (union + eps)
