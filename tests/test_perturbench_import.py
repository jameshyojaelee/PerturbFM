import argparse
import json
from pathlib import Path

from perturbfm.cli import _cmd_data_import_perturbench, _cmd_splits_import_perturbench
from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_store import SplitStore


def test_data_import_perturbench_artifact(tmp_path):
    src_dir = tmp_path / "src"
    ds = make_synthetic_dataset(n_obs=20, n_genes=6, seed=0)
    ds.save_artifact(src_dir)

    out_dir = tmp_path / "out"
    args = argparse.Namespace(dataset=str(src_dir), out=str(out_dir), backed=False)
    _cmd_data_import_perturbench(args)

    loaded = PerturbDataset.load_artifact(out_dir)
    assert loaded.n_obs == ds.n_obs
    assert loaded.n_genes == ds.n_genes


def test_splits_import_perturbench(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    ds = make_synthetic_dataset(n_obs=30, n_genes=5, seed=1)
    ds.save_artifact(data_dir)

    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_idx": list(range(0, 20)),
        "val_idx": list(range(20, 25)),
        "test_idx": list(range(25, 30)),
        "ood_axes": {"context": ["C0"]},
    }
    (split_dir / "official.json").write_text(json.dumps(payload))

    split_store_dir = tmp_path / "split_store"
    monkeypatch.setenv("PERTURBFM_SPLIT_DIR", str(split_store_dir))

    args = argparse.Namespace(dataset=str(data_dir), data=str(data_dir), split_dir=str(split_dir))
    _cmd_splits_import_perturbench(args)

    store = SplitStore.default()
    hashes = store.list()
    assert len(hashes) == 1
    split = store.load(hashes[0])
    assert split.frozen_hash == hashes[0]
    assert split.notes.get("source") == "perturbench"
    assert len(split.train_idx) == 20
