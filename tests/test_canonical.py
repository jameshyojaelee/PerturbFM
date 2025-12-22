import numpy as np

from perturbfm.data.registry import make_synthetic_dataset


def test_required_obs_fields_present():
    ds = make_synthetic_dataset(n_obs=10, n_genes=5, seed=1)
    for field in ("pert_id", "context_id", "batch_id", "is_control"):
        assert field in ds.obs
    ds.validate()


def test_slicing_preserves_alignment():
    ds = make_synthetic_dataset(n_obs=20, n_genes=3, seed=2)
    idx = [0, 3, 5]
    sub = ds.select(idx)
    assert sub.n_obs == len(idx)
    assert sub.X_control.shape[0] == len(idx)
    assert len(sub.obs["pert_id"]) == len(idx)


def test_delta_matches():
    ds = make_synthetic_dataset(n_obs=15, n_genes=4, seed=3)
    assert np.allclose(ds.delta, ds.X_pert - ds.X_control)
