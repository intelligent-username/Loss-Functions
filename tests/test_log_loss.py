import numpy as np
import pytest
from src.log_loss import log_loss


def test_log_loss():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    result = log_loss(y_true, y_pred)
    expected = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)

