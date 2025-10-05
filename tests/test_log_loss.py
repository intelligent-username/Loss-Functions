import numpy as np
import pytest
from sklearn.metrics import log_loss as sk_log_loss

from src.log_loss import log_loss


def test_log_loss():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.1, 0.9, 0.8])
    result = log_loss(y_true, y_pred)
    expected = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)


def test_log_loss_matches_sklearn_case1():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.85, 0.3, 0.7])
    result = log_loss(y_true, y_pred)
    expected = sk_log_loss(y_true, y_pred)
    assert np.isclose(result, expected)


def test_log_loss_matches_sklearn_case2():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.75, 0.2, 0.65, 0.35])
    result = log_loss(y_true, y_pred)
    expected = sk_log_loss(y_true, y_pred)
    assert np.isclose(result, expected)


def test_log_loss_matches_sklearn_case3():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0.15, 0.25, 0.8, 0.9, 0.7])
    result = log_loss(y_true, y_pred)
    expected = sk_log_loss(y_true, y_pred)
    assert np.isclose(result, expected)


def test_log_loss_matches_sklearn_case4():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.05, 0.95, 0.2, 0.8, 0.3, 0.85])
    result = log_loss(y_true, y_pred)
    expected = sk_log_loss(y_true, y_pred)
    assert np.isclose(result, expected)


def test_log_loss_matches_sklearn_case5():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0.4, 0.6, 0.9, 0.1, 0.8])
    result = log_loss(y_true, y_pred)
    expected = sk_log_loss(y_true, y_pred)
    assert np.isclose(result, expected)

