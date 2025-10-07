import numpy as np

def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Categorical Cross-Entropy Loss.

    This loss function measures the difference between true class labels and predicted probabilities.
    Use for multi-class classification problems. It penalizes incorrect predictions more heavily when the
    model is confident, encouraging the model to output accurate probability distributions.

    Attributes:
        y_true (np.ndarray): True class labels. Can be class indices (1D array) or one-hot encoded (2D array).
        y_pred (np.ndarray): Predicted probabilities for each class, shape should match y_true or be (n_samples, n_classes).

    Output:
        float: The average categorical cross-entropy loss across all samples.
    """
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    if y_true.ndim == 1 or y_true.shape != y_pred.shape:
        # Assume y_true contains class indices
        n = y_true.shape[0]
        return -np.mean(np.log(y_pred[np.arange(n), y_true]))
    else:
        # Assume y_true is one-hot encoded
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred)) / n
