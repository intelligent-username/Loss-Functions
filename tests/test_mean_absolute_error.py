import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error as sk_mean_absolute_error

from src.mean_absolute_error import mean_absolute_error


def test_mean_absolute_error():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 2.1, 2.9])
    result = mean_absolute_error(y_true, y_pred)
    expected = np.mean(np.abs(y_true - y_pred))
    assert np.isclose(result, expected)


def test_mean_absolute_error_matches_sklearn_case1():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.1, -0.1, 0.2])
    result = mean_absolute_error(y_true, y_pred)
    expected = sk_mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_absolute_error_matches_sklearn_case2():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.9, 1.9, 3.2, 4.1])
    result = mean_absolute_error(y_true, y_pred)
    expected = sk_mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_absolute_error_matches_sklearn_case3():
    y_true = np.array([-2.0, -1.0, 0.0, 1.0])
    y_pred = np.array([-1.5, -0.4, 0.3, 1.2])
    result = mean_absolute_error(y_true, y_pred)    
    expected = sk_mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_absolute_error_matches_sklearn_case4():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[0.8, 2.1], [3.2, 3.9]])
    result = mean_absolute_error(y_true, y_pred)
    expected = sk_mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected)


def test_mean_absolute_error_matches_sklearn_case5():
    y_true = np.linspace(-1, 1, 5)
    y_pred = np.linspace(-0.5, 1.5, 5)
    result = mean_absolute_error(y_true, y_pred)
    expected = sk_mean_absolute_error(y_true, y_pred)
    assert np.isclose(result, expected)

