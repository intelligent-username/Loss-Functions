import numpy as np
import pytest
from src.categorical_cross_entropy import categorical_cross_entropy


def test_categorical_cross_entropy():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    result = categorical_cross_entropy(y_true, y_pred)
    expected = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    assert np.isclose(result, expected)


def test_categorical_cross_entropy_perfect_predictions():
    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = categorical_cross_entropy(y_true, y_pred)
    # We clip predictions, so loss is small but not zero
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    expected = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=1))
    assert np.isclose(result, expected)


def test_categorical_cross_entropy_single_sample():
    y_true = np.array([[0, 1, 0]])
    y_pred = np.array([[0.2, 0.7, 0.1]])
    result = categorical_cross_entropy(y_true, y_pred)
    expected = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    assert np.isclose(result, expected)


def test_categorical_cross_entropy_uniform_predictions():
    y_true = np.array([[1, 0, 0], [0, 0, 1]])
    y_pred = np.array([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]])
    result = categorical_cross_entropy(y_true, y_pred)
    expected = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    assert np.isclose(result, expected)

