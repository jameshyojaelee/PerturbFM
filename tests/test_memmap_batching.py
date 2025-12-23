import numpy as np

from perturbfm.data.batching import batch_iterator, iter_index_batches
from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.registry import make_synthetic_dataset


def test_memmap_roundtrip(tmp_path):
    ds = make_synthetic_dataset(n_obs=40, n_genes=8, seed=0)
    out_dir = tmp_path / "memmap"
    ds.save_memmap_artifact(out_dir)

    loaded = PerturbDataset.load_artifact(out_dir)
    assert loaded.n_obs == ds.n_obs
    assert loaded.n_genes == ds.n_genes
    assert np.allclose(loaded.delta[:5], ds.delta[:5])


def test_iter_index_batches_deterministic():
    idx = np.arange(12)
    batches_a = [b.tolist() for b in iter_index_batches(idx, batch_size=4, seed=0)]
    batches_b = [b.tolist() for b in iter_index_batches(idx, batch_size=4, seed=0)]
    assert batches_a == batches_b
    flat = [i for batch in batches_a for i in batch]
    assert sorted(flat) == idx.tolist()


def test_batch_iterator_fields(tmp_path):
    ds = make_synthetic_dataset(n_obs=10, n_genes=4, seed=1)
    batches = list(batch_iterator(ds, np.arange(ds.n_obs), batch_size=5, seed=0))
    assert len(batches) == 2
    for batch in batches:
        assert "x_control" in batch
        assert "delta" in batch
        assert "pert_id" in batch
        assert "context_id" in batch
