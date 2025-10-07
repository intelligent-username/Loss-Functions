import numpy as np


def negative_log_likelihood(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """
    Negative Log Likelihood.

    This loss function measures how well the predicted probability distribution matches the true
    class labels. It penalizes the model for assigning low probabilities to correct classes,
    encouraging confident and accurate predictions in classification tasks.

    Attributes:
        y_true (np.ndarray): True class indices.
        probabilities (np.ndarray): Predicted probabilities for each class, shape (n_samples, n_classes).

    Output:
        float: The average negative log likelihood across all samples.
    """
    n = len(y_true)
    epsilon = 1e-15  # To avoid log(0)

    loss = -np.sum(np.log(np.clip(probabilities, epsilon, 1 - epsilon))[np.arange(n), y_true]) / n
    return float(loss)
