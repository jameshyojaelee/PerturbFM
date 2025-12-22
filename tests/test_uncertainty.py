import numpy as np

from perturbfm.eval.uncertainty_metrics import compute_uncertainty_metrics


def test_uncertainty_coverage_and_nll():
    y_true = np.array([[0.0, 0.0], [1.0, 1.0]])
    mean = y_true.copy()
    var = np.ones_like(mean)
    metrics = compute_uncertainty_metrics(y_true, mean, var)
    for key, val in metrics["coverage"].items():
        assert val == 1.0
    assert metrics["nll"] == 0.0
