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


def test_conformal_uses_calib_idx(monkeypatch, tmp_path):
    from perturbfm.data.registry import make_synthetic_dataset
    from perturbfm.data.splits.split_spec import context_ood_split
    from perturbfm.data.splits.split_store import SplitStore
    from perturbfm import eval as eval_pkg

    ds = make_synthetic_dataset(n_obs=30, n_genes=5, seed=2)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)
    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=1, val_fraction=0.4).freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(tmp_path / "splits"))

    captured = {}

    def _capture(residuals, alphas):
        captured["shape"] = residuals.shape
        return {"0.9": residuals.mean(axis=0).tolist()}

    monkeypatch.setattr(eval_pkg.evaluator, "conformal_intervals", _capture)

    eval_pkg.evaluator.run_baseline(
        data_path=str(data_dir),
        split_hash=split.frozen_hash,
        baseline_name="global_mean",
        conformal=True,
    )

    calib_idx, _ = eval_pkg.evaluator._get_calib_idx(split)
    assert captured["shape"][0] == len(calib_idx)
