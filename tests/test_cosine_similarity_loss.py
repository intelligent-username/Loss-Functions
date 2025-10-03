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

