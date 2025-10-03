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

