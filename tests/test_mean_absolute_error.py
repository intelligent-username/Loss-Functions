import numpy as np
import pytest
from src.mean_absolute_error import mean_absolute_error


def test_mean_absolute_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = mean_absolute_error(y_true, y_pred)
    expected = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(result, expected)

