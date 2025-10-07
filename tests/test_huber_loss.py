import numpy as np
import pytest
from src.huber_loss import huber_loss


def test_huber_loss():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    delta = 1.0
    result = huber_loss(y_true, y_pred, delta)
    diff = y_true - y_pred
    expected = np.mean(np.where(np.abs(diff) <= delta, 0.5 * diff**2, delta * np.abs(diff) - 0.5 * delta**2))
    assert np.isclose(result, expected)


def test_huber_loss_delta_zero():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 1.8])
    delta = 0.1  # Small delta to test near-MSE behavior
    result = huber_loss(y_true, y_pred, delta)
    diff = y_true - y_pred
    expected = np.mean(np.where(np.abs(diff) <= delta, 0.5 * diff**2, delta * np.abs(diff) - 0.5 * delta**2))
    assert np.isclose(result, expected)


def test_huber_loss_large_delta():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    delta = 10.0
    result = huber_loss(y_true, y_pred, delta)
    diff = y_true - y_pred
    expected = np.mean(np.where(np.abs(diff) <= delta, 0.5 * diff**2, delta * np.abs(diff) - 0.5 * delta**2))
    assert np.isclose(result, expected)


def test_huber_loss_single_sample():
    y_true = np.array([2.0])
    y_pred = np.array([1.5])
    delta = 1.0
    result = huber_loss(y_true, y_pred, delta)
    diff = y_true - y_pred
    expected = np.mean(np.where(np.abs(diff) <= delta, 0.5 * diff**2, delta * np.abs(diff) - 0.5 * delta**2))
    assert np.isclose(result, expected)

