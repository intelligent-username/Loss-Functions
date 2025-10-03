import numpy as np
import pytest
from src.negative_log_likelihood import negative_log_likelihood


def test_negative_log_likelihood():
    y_true = np.array([0, 1, 2])
    probabilities = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    result = negative_log_likelihood(y_true, probabilities)
    expected = -np.mean(np.log(probabilities[np.arange(len(y_true)), y_true]))
    assert np.isclose(result, expected)

