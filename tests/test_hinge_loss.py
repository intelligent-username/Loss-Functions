import numpy as np
import pytest
from src.hinge_loss import hinge_loss


def test_hinge_loss():
    y_true = np.array([-1, 1, 1])
    y_pred = np.array([-0.5, 0.8, 1.2])
    result = hinge_loss(y_true, y_pred)
    expected = np.mean(np.maximum(0, 1 - y_true * y_pred))
    assert np.isclose(result, expected)


def test_hinge_loss_perfect_predictions():
    y_true = np.array([-1, 1, 1])
    y_pred = np.array([-2.0, 2.0, 2.0])
    result = hinge_loss(y_true, y_pred)
    expected = 0.0  # Perfect predictions should give zero loss
    assert np.isclose(result, expected)


def test_hinge_loss_all_wrong():
    y_true = np.array([-1, 1, 1])
    y_pred = np.array([0.5, -0.8, -1.2])
    result = hinge_loss(y_true, y_pred)
    expected = np.mean(np.maximum(0, 1 - y_true * y_pred))
    assert np.isclose(result, expected)


def test_hinge_loss_single_sample():
    y_true = np.array([1])
    y_pred = np.array([0.5])
    result = hinge_loss(y_true, y_pred)
    expected = np.mean(np.maximum(0, 1 - y_true * y_pred))
    assert np.isclose(result, expected)

