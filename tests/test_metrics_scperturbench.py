import numpy as np

from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics


def test_scperturbench_identity_vs_collapsed():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(50, 10))
    y_pred_identity = y_true.copy()
    y_pred_collapsed = np.zeros_like(y_true)
    obs = {"pert_id": ["P0"] * 50, "context_id": ["C0"] * 50}

    m_id = compute_scperturbench_metrics(y_true, y_pred_identity, obs)["global"]
    m_col = compute_scperturbench_metrics(y_true, y_pred_collapsed, obs)["global"]

    assert m_id["MSE"] < m_col["MSE"]
    # collapsed predictor yields undefined PCC (nan)
    assert np.isfinite(m_id["PCC_delta"])
    assert np.isnan(m_col["PCC_delta"])
    assert m_id["Energy"] < m_col["Energy"]
