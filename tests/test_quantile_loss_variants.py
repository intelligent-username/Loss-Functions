import numpy as np
import pytest
from src.quantile_loss import quantile_loss


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.9])
def test_quantile_loss_tau_variants(tau):
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([0.9, 2.5, 2.5, 5.0])
    diff = y_true - y_pred
    expected_vector = np.where(diff > 0, tau * diff, (1 - tau) * (-diff))
    expected = np.mean(expected_vector)
    assert np.isclose(quantile_loss(y_true, y_pred, tau), expected)


def test_quantile_loss_under_vs_over():
    y_true = np.array([2.0, 2.0])
    y_pred_under = np.array([1.0, 1.5])  # model under-predicts
    y_pred_over = np.array([3.0, 2.5])   # model over-predicts
    tau = 0.7
    loss_under = quantile_loss(y_true, y_pred_under, tau)
    loss_over = quantile_loss(y_true, y_pred_over, tau)
    assert loss_under > loss_over  # with tau>0.5 under-predictions penalized more
