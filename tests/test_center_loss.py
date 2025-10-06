import numpy as np
import pytest
from src.centre_loss import centre_loss


def test_centre_loss():
    features = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    centres = np.array([[1.1, 2.1], [3.1, 4.1]])
    labels = np.array([0, 1, 1])
    result = centre_loss(features, centres, labels)
    expected = 0.5 * np.mean(np.sum((features - centres[labels])**2, axis=1))
    assert np.isclose(result, expected)

