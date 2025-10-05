import numpy as np
import pytest
from sklearn.metrics import mean_squared_error as sk_mean_squared_error

from src.mean_squared_error import mean_squared_error


def test_mean_squared_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = mean_squared_error(y_true, y_pred)
    expected = np.mean((y_true - y_pred) ** 2)
    assert np.isclose(result, expected)


def test_mean_squared_error_matches_sklearn_case1():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.1, -0.1, 0.2])
    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_squared_error_matches_sklearn_case2():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.9, 3.1, 3.8])
    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_squared_error_matches_sklearn_case3():
    y_true = np.array([-2.0, -1.0, 0.0, 1.0])
    y_pred = np.array([-1.5, -0.5, 0.2, 0.9])
    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_squared_error_matches_sklearn_case4():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[0.9, 1.8], [3.1, 4.2]])
    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_squared_error_matches_sklearn_case5():
    y_true = np.linspace(0, 1, 5)
    y_pred = np.linspace(0.1, 1.1, 5)
    result = mean_squared_error(y_true, y_pred)
    expected = sk_mean_squared_error(y_true, y_pred)
    assert np.isclose(result, expected)

