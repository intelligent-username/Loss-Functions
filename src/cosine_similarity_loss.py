import numpy as np


def cosine_similarity_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Cosine Similarity Loss.

    This loss function measures the angular difference between two vectors, emphasizing directional
    alignment over magnitude. It calculates 1 minus the cosine similarity, where perfect alignment
    gives a loss of 0, and opposite directions give a higher loss.

    Attributes:
        y_true (np.ndarray): The true vector representation.
        y_pred (np.ndarray): The predicted vector representation.

    Output:
        float: The cosine similarity loss, ranging from 0 (perfect alignment) to 2 (opposite directions).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for cosine similarity loss.")
    denom = np.linalg.norm(y_true) * np.linalg.norm(y_pred)
    if denom == 0:
        return float("nan")
    cos_sim = np.dot(y_true.flatten(), y_pred.flatten()) / denom
    return 1 - cos_sim
