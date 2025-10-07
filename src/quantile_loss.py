import numpy as np


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Quantile Loss.

    This loss function enables asymmetric penalization of prediction errors based on a quantile
    parameter. It allows biasing predictions toward over- or under-estimation depending on the
    tau value, making it useful for quantile regression and scenarios requiring specific error preferences.

    Attributes:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        tau (float): Quantile parameter between 0 and 1, controlling the asymmetry of the loss.

    Output:
        float: The mean quantile loss across all samples.
    """
    if not (0 < tau < 1):
        raise ValueError("tau must be between 0 and 1 (exclusive).")
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes must match for quantile_loss.")
    diff = y_true - y_pred
    # Under-prediction (y_true > y_pred) -> positive diff
    loss = np.where(diff > 0, tau * diff, (1 - tau) * (-diff))
    return float(np.mean(loss))
