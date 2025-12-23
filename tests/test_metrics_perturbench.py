import numpy as np

from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics


def test_perturbench_rank_and_collapse():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=(40, 12))
    y_pred_perfect = y_true.copy()
    y_pred_noisy = y_true + rng.normal(scale=0.5, size=y_true.shape)
    y_pred_collapsed = np.zeros_like(y_true)

    obs = {"pert_id": ["P0"] * 40, "context_id": ["C0"] * 40}

    m_perf = compute_perturbench_metrics(y_true, y_pred_perfect, obs)["global"]
    m_noisy = compute_perturbench_metrics(y_true, y_pred_noisy, obs)["global"]
    m_col = compute_perturbench_metrics(y_true, y_pred_collapsed, obs)["global"]

    assert m_perf["RankMetrics"] > m_noisy["RankMetrics"]
    assert m_noisy["RankMetrics"] > m_col["RankMetrics"]
    assert m_perf["VarianceDiagnostics"] > m_col["VarianceDiagnostics"]
