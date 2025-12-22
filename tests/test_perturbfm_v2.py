import pytest

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.train.trainer import fit_predict_perturbfm_v2


def test_perturbfm_v2_smoke():
    torch = pytest.importorskip("torch")
    ds = make_synthetic_dataset(n_obs=20, n_genes=8, n_contexts=3, seed=1)
    split = context_ood_split(ds.obs["context_id"], ["C0"], seed=0, val_fraction=0.2).freeze()
    out = fit_predict_perturbfm_v2(ds, split, hidden_dim=16, lr=1e-2, epochs=2, device="cpu", num_bases=2)
    assert out["mean"].shape == out["var"].shape
    assert out["mean"].shape[0] == len(out["idx"])
