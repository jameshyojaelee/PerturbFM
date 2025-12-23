import json

from perturbfm.data.adapters.scperturbench import ScPerturBenchAdapter
from perturbfm.data.registry import make_synthetic_dataset


def test_scperturbench_adapter_load_artifact(tmp_path):
    ds = make_synthetic_dataset(n_obs=15, n_genes=5, seed=0)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)

    adapter = ScPerturBenchAdapter(data_dir)
    loaded = adapter.load()
    assert loaded.n_obs == ds.n_obs
    assert loaded.n_genes == ds.n_genes


def test_scperturbench_adapter_splits(tmp_path):
    split_dir = tmp_path / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    payload = {"train_idx": [0, 1], "val_idx": [2], "test_idx": [3]}
    (split_dir / "split.json").write_text(json.dumps(payload))

    ds = make_synthetic_dataset(n_obs=5, n_genes=4, seed=1)
    data_dir = tmp_path / "data"
    ds.save_artifact(data_dir)

    adapter = ScPerturBenchAdapter(data_dir)
    splits = adapter.load_official_splits(split_dir)
    assert "split" in splits
