import pytest

from perturbfm.eval.evaluator import _require_metrics_complete


def test_require_metrics_complete_missing_panel():
    metrics = {}
    with pytest.raises(ValueError):
        _require_metrics_complete(metrics)


def test_require_metrics_complete_missing_metric():
    metrics = {
        "scperturbench": {"global": {"MSE": 1.0}},
        "perturbench": {"global": {"RMSE": 1.0}},
        "uncertainty": {"coverage": {}, "nll": 0.0, "risk_coverage": {}, "ood_auroc": None},
    }
    with pytest.raises(ValueError):
        _require_metrics_complete(metrics)


def test_require_metrics_complete_ok():
    metrics = {
        "scperturbench": {"global": {"MSE": 0, "PCC_delta": 0, "Energy": 0, "Wasserstein": 0, "KL": 0, "Common_DEGs": 0}},
        "perturbench": {"global": {"RMSE": 0, "RankMetrics": 0, "VarianceDiagnostics": 0}},
        "uncertainty": {"coverage": {}, "nll": 0, "risk_coverage": {}, "ood_auroc": None},
    }
    _require_metrics_complete(metrics)
