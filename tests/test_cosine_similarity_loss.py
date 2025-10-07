import numpy as np
import pytest
from src.cosine_similarity_loss import cosine_similarity_loss


def test_cosine_similarity_loss():
    y_true = np.array([[1.0, 0.0], [0.0, 1.0]])
    y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
    result = cosine_similarity_loss(y_true, y_pred)
    cos_sim = (y_true * y_pred).sum(axis=1) / (np.linalg.norm(y_true, axis=1) * np.linalg.norm(y_pred, axis=1))
    expected = 1 - np.mean(cos_sim)
    assert np.isclose(result, expected)


def test_cosine_similarity_loss_identical_vectors():
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = cosine_similarity_loss(y_true, y_pred)
    expected = 0.0  # Identical vectors should have zero loss
    assert np.isclose(result, expected)


def test_cosine_similarity_loss_orthogonal_vectors():
    y_true = np.array([[1.0, 0.0]])
    y_pred = np.array([[0.0, 1.0]])
    result = cosine_similarity_loss(y_true, y_pred)
    expected = 1.0  # Orthogonal vectors have cosine similarity 0, so loss is 1
    assert np.isclose(result, expected)


def test_cosine_similarity_loss_single_dimension():
    y_true = np.array([[1.0]])
    y_pred = np.array([[0.8]])
    result = cosine_similarity_loss(y_true, y_pred)
    expected = 0.0  # Cosine similarity is 1.0, loss is 0.0
    assert np.isclose(result, expected)

