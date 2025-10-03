import numpy as np
import pytest
from src.categorical_cross_entropy import categorical_cross_entropy


def test_categorical_cross_entropy():
    y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    result = categorical_cross_entropy(y_true, y_pred)
    expected = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
    assert np.isclose(result, expected)

