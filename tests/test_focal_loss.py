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


def test_focal_loss_perfect_predictions():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.0, 1.0, 1.0])
    alpha = 0.25
    gamma = 2.0
    result = focal_loss(y_true, y_pred, alpha, gamma)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    expected = -np.mean(alpha * (1 - y_pred)**gamma * y_true * np.log(y_pred) + (1 - alpha) * y_pred**gamma * (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)


def test_focal_loss_different_alpha_gamma():
    y_true = np.array([1, 0])
    y_pred = np.array([0.7, 0.3])
    alpha = 0.5
    gamma = 1.0
    result = focal_loss(y_true, y_pred, alpha, gamma)
    expected = -np.mean(alpha * (1 - y_pred)**gamma * y_true * np.log(y_pred) + (1 - alpha) * y_pred**gamma * (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)


def test_focal_loss_single_sample():
    y_true = np.array([1])
    y_pred = np.array([0.8])
    alpha = 0.25
    gamma = 2.0
    result = focal_loss(y_true, y_pred, alpha, gamma)
    expected = -np.mean(alpha * (1 - y_pred)**gamma * y_true * np.log(y_pred) + (1 - alpha) * y_pred**gamma * (1 - y_true) * np.log(1 - y_pred))
    assert np.isclose(result, expected)

