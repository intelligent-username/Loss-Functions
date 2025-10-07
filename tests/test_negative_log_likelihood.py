import numpy as np
import pytest
from src.negative_log_likelihood import negative_log_likelihood


def test_negative_log_likelihood():
    y_true = np.array([0, 1, 2])
    probabilities = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    result = negative_log_likelihood(y_true, probabilities)
    expected = -np.mean(np.log(probabilities[np.arange(len(y_true)), y_true]))
    assert np.isclose(result, expected)


def test_negative_log_likelihood_perfect_predictions():
    y_true = np.array([0, 1])
    probabilities = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = negative_log_likelihood(y_true, probabilities)
    # Perfect predictions should give very small loss due to clipping
    epsilon = 1e-15
    probabilities_clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    expected = -np.mean(np.log(probabilities_clipped[np.arange(len(y_true)), y_true]))
    assert np.isclose(result, expected)


def test_negative_log_likelihood_single_sample():
    y_true = np.array([1])
    probabilities = np.array([[0.2, 0.7, 0.1]])
    result = negative_log_likelihood(y_true, probabilities)
    expected = -np.mean(np.log(probabilities[np.arange(len(y_true)), y_true]))
    assert np.isclose(result, expected)


def test_negative_log_likelihood_different_classes():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6]])
    result = negative_log_likelihood(y_true, probabilities)
    expected = -np.mean(np.log(probabilities[np.arange(len(y_true)), y_true]))
    assert np.isclose(result, expected)

