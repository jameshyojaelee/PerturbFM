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


def test_v1_enforces_metric_completeness(monkeypatch, tmp_path):
    import numpy as np

    from perturbfm.data.registry import make_synthetic_dataset
    from perturbfm.data.splits.split_spec import context_ood_split
    from perturbfm.data.splits.split_store import SplitStore
    from perturbfm.eval import evaluator

    ds = make_synthetic_dataset(n_obs=20, n_genes=6, seed=1)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)
    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0).freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(tmp_path / "splits"))

    # patch metrics to be incomplete
    monkeypatch.setattr(evaluator, "compute_scperturbench_metrics", lambda *a, **k: {"global": {"MSE": 0.0}})
    monkeypatch.setattr(evaluator, "compute_perturbench_metrics", lambda *a, **k: {"global": {"RMSE": 0.0}})
    monkeypatch.setattr(evaluator, "compute_uncertainty_metrics", lambda *a, **k: {"coverage": {}, "nll": 0.0, "risk_coverage": {}, "ood_auroc": None})

    adj = np.eye(ds.n_genes, dtype=np.float32)
    pert_gene_masks = {pid: np.zeros(ds.n_genes, dtype=np.float32) for pid in set(ds.obs["pert_id"])}

    with pytest.raises(ValueError):
        evaluator.run_perturbfm_v1(
            data_path=str(data_dir),
            split_hash=split.frozen_hash,
            adjacency=adj,
            pert_gene_masks=pert_gene_masks,
            epochs=1,
            hidden_dim=8,
        )


def _patch_incomplete_metrics(monkeypatch, evaluator):
    monkeypatch.setattr(evaluator, "compute_scperturbench_metrics", lambda *a, **k: {"global": {"MSE": 0.0}})
    monkeypatch.setattr(evaluator, "compute_perturbench_metrics", lambda *a, **k: {"global": {"RMSE": 0.0}})
    monkeypatch.setattr(evaluator, "compute_uncertainty_metrics", lambda *a, **k: {"coverage": {}, "nll": 0.0, "risk_coverage": {}, "ood_auroc": None})


def test_v0_enforces_metric_completeness(monkeypatch, tmp_path):
    from perturbfm.data.registry import make_synthetic_dataset
    from perturbfm.data.splits.split_spec import context_ood_split
    from perturbfm.data.splits.split_store import SplitStore
    from perturbfm.eval import evaluator

    ds = make_synthetic_dataset(n_obs=20, n_genes=6, seed=1)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)
    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0).freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(tmp_path / "splits"))

    _patch_incomplete_metrics(monkeypatch, evaluator)
    with pytest.raises(ValueError):
        evaluator.run_perturbfm_v0(
            data_path=str(data_dir),
            split_hash=split.frozen_hash,
            epochs=1,
            hidden_dim=8,
        )


def test_v2_enforces_metric_completeness(monkeypatch, tmp_path):
    from perturbfm.data.registry import make_synthetic_dataset
    from perturbfm.data.splits.split_spec import context_ood_split
    from perturbfm.data.splits.split_store import SplitStore
    from perturbfm.eval import evaluator

    ds = make_synthetic_dataset(n_obs=20, n_genes=6, seed=1)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)
    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0).freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(tmp_path / "splits"))

    _patch_incomplete_metrics(monkeypatch, evaluator)
    with pytest.raises(ValueError):
        evaluator.run_perturbfm_v2(
            data_path=str(data_dir),
            split_hash=split.frozen_hash,
            adjacency=None,
            epochs=1,
            hidden_dim=8,
        )
