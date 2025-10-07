import numpy as np


def cosine_similarities_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Batched Cosine Similarity Loss.

    Extension of cosine similarity loss to batches of vector pairs. Computes the cosine similarity
    for each pair of corresponding vectors and returns 1 minus the average similarity, focusing on
    directional alignment while ignoring magnitude.

    Attributes:
        y_true (np.ndarray): True vectors, shape (n_samples, n_features).
        y_pred (np.ndarray): Predicted vectors, same shape as y_true.

    Output:
        float: The cosine similarity loss averaged across the batch (0 = identical direction).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for batched cosine similarity loss.")
    if y_true.ndim != 2:
        raise ValueError("Inputs must be 2D arrays (n_samples, n_features).")
    norms_true = np.linalg.norm(y_true, axis=1)
    norms_pred = np.linalg.norm(y_pred, axis=1)
    denom = norms_true * norms_pred
    # Avoid divide by zero
    valid = denom > 0
    cos_sim = np.zeros_like(denom)
    cos_sim[valid] = np.sum(y_true[valid] * y_pred[valid], axis=1) / denom[valid]
    return 1 - float(np.mean(cos_sim))
