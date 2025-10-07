import numpy as np
import pytest
from src.cosine_similarities_loss import cosine_similarities_loss


def test_cosine_similarities_basic():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.95]])
    dots = np.sum(y_true * y_pred, axis=1)
    magn = np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1)
    expected = 1 - np.mean(dots / magn)
    assert np.isclose(cosine_similarities_loss(y_true, y_pred), expected)


def test_cosine_similarities_identical():
    y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y_pred = y_true.copy()
    assert np.isclose(cosine_similarities_loss(y_true, y_pred), 0.0)


def test_cosine_similarities_opposite():
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[-1.0, 0.0]])
    assert np.isclose(cosine_similarities_loss(y_true, y_pred), 2.0)


def test_cosine_similarities_zero_vector():
    y_true = np.array([[0.0, 0.0]])
    y_pred = np.array([[1.0, 0.0]])
    loss = cosine_similarities_loss(y_true, y_pred)
    assert np.isnan(loss) or np.isfinite(loss)
