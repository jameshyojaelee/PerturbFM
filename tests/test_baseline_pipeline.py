from pathlib import Path

import numpy as np

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.eval.evaluator import run_baseline


def test_baseline_pipeline(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    ds = make_synthetic_dataset(n_obs=50, n_genes=8, n_contexts=3, seed=0)
    ds.save_artifact(data_dir)

    split = context_ood_split(ds.obs["context_id"], ["C0"], seed=0, val_fraction=0.2)
    split.freeze()
    split_store_dir = tmp_path / "splits"
    store = SplitStore(root=split_store_dir)
    store.save(split)

    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(split_store_dir))
    run_dir = run_baseline(
        data_path=str(data_dir),
        split_hash=split.frozen_hash,
        baseline_name="global_mean",
        out_dir=str(tmp_path / "run"),
    )

    pred_path = Path(run_dir) / "predictions.npz"
    assert pred_path.exists()
    npz = np.load(pred_path)
    mean = npz["mean"]
    var = npz["var"]
    idx = npz["idx"]
    assert mean.shape == var.shape
    assert mean.shape[0] == len(idx)

    for name in ("metrics.json", "calibration.json", "report.html", "config.json", "split_hash.txt"):
        assert (Path(run_dir) / name).exists()
