import numpy as np
import pytest
from src.quantile_loss import quantile_loss


def test_quantile_loss():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    tau = 0.5
    result = quantile_loss(y_true, y_pred, tau)
    expected = np.mean(np.maximum(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true)))
    assert np.isclose(result, expected)

