from pathlib import Path

import numpy as np
import pytest

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.eval.evaluator import run_baseline


def test_eval_split_val_uses_val_idx(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    ds = make_synthetic_dataset(n_obs=50, n_genes=8, n_contexts=3, seed=0)
    ds.save_artifact(data_dir)

    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0, val_fraction=0.2)
    split.freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(store.root))

    run_dir = run_baseline(
        data_path=str(data_dir),
        split_hash=split.frozen_hash,
        baseline_name="global_mean",
        out_dir=str(tmp_path / "run"),
        eval_split="val",
    )

    npz = np.load(Path(run_dir) / "predictions.npz")
    idx = npz["idx"]
    assert set(idx.tolist()) == set(split.val_idx.tolist())


def test_eval_split_val_disallows_conformal(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    ds = make_synthetic_dataset(n_obs=30, n_genes=6, n_contexts=2, seed=1)
    ds.save_artifact(data_dir)

    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=1, val_fraction=0.2)
    split.freeze()
    store = SplitStore(root=tmp_path / "splits")
    store.save(split)
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(store.root))

    with pytest.raises(ValueError):
        run_baseline(
            data_path=str(data_dir),
            split_hash=split.frozen_hash,
            baseline_name="global_mean",
            out_dir=str(tmp_path / "run2"),
            eval_split="val",
            conformal=True,
        )
