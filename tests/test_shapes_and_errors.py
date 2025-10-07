import numpy as np
import pytest
from src import (
    mean_squared_error,
    mean_absolute_error,
    huber_loss,
    quantile_loss,
    cosine_similarity_loss,
    cosine_similarities_loss,
)


def test_shape_mismatch_mse():
    with pytest.raises(ValueError):
        mean_squared_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_shape_mismatch_mae():
    with pytest.raises(ValueError):
        mean_absolute_error(np.array([1, 2, 3]), np.array([1, 2]))


def test_shape_mismatch_cosine():
    with pytest.raises(ValueError):
        cosine_similarity_loss(np.array([1, 0]), np.array([1, 0, 0]))


def test_shape_mismatch_batched_cosine():
    with pytest.raises(ValueError):
        cosine_similarities_loss(np.array([[1, 0], [0, 1]]), np.array([[1, 0]]))


def test_invalid_tau_quantile():
    with pytest.raises(ValueError):
        quantile_loss(np.array([1, 2]), np.array([1, 2]), tau=1.5)


def test_negative_delta_huber():
    with pytest.raises(ValueError):
        huber_loss(np.array([1, 2]), np.array([1, 2]), delta=-1.0)
