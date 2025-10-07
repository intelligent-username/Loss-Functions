import numpy as np
import pytest
from src.huber_loss import huber_loss


def test_huber_loss_zero_difference():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = y_true.copy()
    assert huber_loss(y_true, y_pred, delta=1.0) == 0.0


def test_huber_loss_large_delta_equals_mse():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    delta = 10.0
    # With very large delta all errors fall in quadratic region
    diff = y_true - y_pred
    expected = np.mean(0.5 * diff**2)
    assert np.isclose(huber_loss(y_true, y_pred, delta), expected)


def test_huber_loss_small_delta_approaches_mae():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.2, 1.7, 2.4])
    delta = 1e-6
    diff = y_true - y_pred
    expected = np.mean(delta * np.abs(diff) - 0.5 * delta**2)
    assert np.isclose(huber_loss(y_true, y_pred, delta), expected)
