import numpy as np
import pytest
from src.focal_loss import focal_loss


def test_focal_loss():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    alpha = 0.25
    gamma = 2.0
    result = focal_loss(y_true, y_pred, alpha, gamma)
    expected = -np.mean(alpha * (1 - y_pred)**gamma * y_true * np.log(y_pred) + (1 - alpha) * y_pred**gamma * (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)

