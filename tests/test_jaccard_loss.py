import numpy as np
import pytest
from src.jaccard_loss import jaccard_loss


def test_jaccard_loss():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 1, 0])
    result = jaccard_loss(y_true, y_pred)
    intersection = np.sum(y_true & y_pred)
    union = np.sum(y_true | y_pred)
    expected = 1 - intersection / union if union > 0 else 0
    assert np.isclose(result, expected)


def test_jaccard_loss_perfect_match():
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 1])
    result = jaccard_loss(y_true, y_pred)
    expected = 0.0  # Perfect match -> zero loss
    assert np.isclose(result, expected)


def test_jaccard_loss_no_overlap():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 1])
    result = jaccard_loss(y_true, y_pred)
    expected = 1.0  # No overlap -> loss of 1
    assert np.isclose(result, expected)


def test_jaccard_loss_single_element():
    y_true = np.array([1])
    y_pred = np.array([1])
    result = jaccard_loss(y_true, y_pred)
    expected = 0.0  # Single match
    assert np.isclose(result, expected)

