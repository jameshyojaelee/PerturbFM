import numpy as np

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.train.trainer import fit_predict_baseline, get_baseline


def test_latent_shift_beats_control_only():
    ds = make_synthetic_dataset(n_obs=80, n_genes=16, seed=0)
    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0, val_fraction=0.2)

    control_model = get_baseline("control_only")
    control_pred = fit_predict_baseline(control_model, ds, split)
    test_idx = split.test_idx
    is_control = np.asarray(ds.obs["is_control"])[test_idx]
    non_control_idx = test_idx[~is_control]
    control_mse = np.mean((control_pred["mean"][~is_control] - ds.delta[non_control_idx]) ** 2)

    latent_model = get_baseline("latent_shift", n_components=ds.n_genes)
    latent_pred = fit_predict_baseline(latent_model, ds, split)
    latent_mse = np.mean((latent_pred["mean"][~is_control] - ds.delta[non_control_idx]) ** 2)

    assert latent_mse < control_mse
