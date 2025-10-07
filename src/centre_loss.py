import numpy as np


def centre_loss(
    features: np.ndarray,
    centres: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Centre Loss.

    This loss function encourages features of the same class to cluster together in feature space.
    It measures the squared Euclidean distance between each feature vector and its corresponding
    class center, helping to reduce intra-class variance and improve feature learning.

    Note: A 0.5 scaling factor is applied.

    Attributes:
        features (np.ndarray): Feature representations of the samples, shape (n_samples, n_features).
        centres (np.ndarray): Class centers, shape (n_classes, n_features).
        labels (np.ndarray): Class labels for each sample, shape (n_samples,).

    Output:
        float: The average centre loss across all samples.
    """
    diffs = features - centres[labels]
    return float(0.5 * np.mean(np.sum(diffs**2, axis=1)))
