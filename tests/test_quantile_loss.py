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


def test_quantile_loss_tau_zero():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.8])
    tau = 0.01  # Very small tau to test near-underestimation penalty
    result = quantile_loss(y_true, y_pred, tau)
    expected = np.mean(np.maximum(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true)))
    assert np.isclose(result, expected)


def test_quantile_loss_tau_one():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.8])
    tau = 0.99  # Very large tau to test near-overestimation penalty
    result = quantile_loss(y_true, y_pred, tau)
    expected = np.mean(np.maximum(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true)))
    assert np.isclose(result, expected)


def test_quantile_loss_single_sample():
    y_true = np.array([2.0])
    y_pred = np.array([1.5])
    tau = 0.5
    result = quantile_loss(y_true, y_pred, tau)
    expected = np.mean(np.maximum(tau * (y_true - y_pred), (1 - tau) * (y_pred - y_true)))
    assert np.isclose(result, expected)

