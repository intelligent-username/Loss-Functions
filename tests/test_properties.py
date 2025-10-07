import numpy as np
from src import cosine_similarity_loss, cosine_similarities_loss, mean_absolute_error, mean_squared_error


def test_cosine_similarity_symmetry_single():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([3.0, 2.0, 1.0])
    loss_ab = cosine_similarity_loss(a, b)
    loss_ba = cosine_similarity_loss(b, a)
    assert np.isclose(loss_ab, loss_ba)


def test_cosine_similarities_batch_equivalence():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[0.9, 0.1], [0.1, 0.95]])
    single_losses = [cosine_similarity_loss(ai, bi) for ai, bi in zip(a, b)]
    batch_loss = cosine_similarities_loss(a, b)
    assert 0 <= batch_loss <= 2
    assert np.isclose(batch_loss, np.mean(single_losses), atol=1e-6)


def test_mae_non_negative():
    x = np.array([1.0, -2.0, 3.5])
    y = np.array([0.5, -1.5, 3.0])
    assert mean_absolute_error(x, y) >= 0


def test_mse_non_negative():
    x = np.array([1.0, -2.0, 3.5])
    y = np.array([0.5, -1.5, 3.0])
    assert mean_squared_error(x, y) >= 0
