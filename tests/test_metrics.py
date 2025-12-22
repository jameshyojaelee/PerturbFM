import numpy as np

from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics


def test_scperturbench_mse_pcc():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_pred = np.array([[0.0, 0.0], [2.0, 2.0]])
    obs = {"pert_id": ["P0", "P0"], "context_id": ["C0", "C0"]}
    metrics = compute_scperturbench_metrics(y_true, y_pred, obs)
    assert "MSE" in metrics["global"]
    assert metrics["global"]["MSE"] == np.mean((y_true - y_pred) ** 2)
    assert "PCC_delta" in metrics["global"]


def test_perturbench_rmse():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_pred = np.array([[0.0, 0.0], [2.0, 2.0]])
    obs = {"pert_id": ["P0", "P0"], "context_id": ["C0", "C0"]}
    metrics = compute_perturbench_metrics(y_true, y_pred, obs)
    expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
    assert metrics["global"]["RMSE"] == expected
