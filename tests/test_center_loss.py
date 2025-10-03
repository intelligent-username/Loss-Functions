import numpy as np
import pytest
from src.center_loss import center_loss


def test_center_loss():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    centers = np.array([[1.1, 2.1], [3.1, 4.1]])
    labels = np.array([0, 1, 1])
    result = center_loss(features, centers, labels)
    expected = 0.5 * np.mean(np.sum((features - centers[labels])**2, axis=1))
    assert np.isclose(result, expected)

