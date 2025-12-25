import numpy as np
import pytest

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import Split
from perturbfm.models.baselines.additive_mean import AdditiveMeanBaseline
from perturbfm.train.trainer import fit_predict_perturbfm_v3_residual


def _make_residual_dataset() -> PerturbDataset:
    delta = np.array(
        [
            [1.0, 0.0, 0.0],  # A
            [0.0, 1.0, 0.0],  # B
            [0.0, 0.0, 1.0],  # C
            [1.5, 1.5, 0.5],  # A+B (train)
            [1.5, 1.5, 0.5],  # A+B (test replicate)
        ],
        dtype=np.float32,
    )
    X_control = np.zeros_like(delta, dtype=np.float32)
    obs = {
        "pert_id": ["A", "B", "C", "A+B", "A+B"],
        "context_id": ["C0"] * 5,
        "batch_id": ["B0"] * 5,
        "is_control": [False] * 5,
        "pert_genes": [["A"], ["B"], ["C"], ["A", "B"], ["A", "B"]],
    }
    var = ["A", "B", "C"]
    ds = PerturbDataset(X_control=X_control, X_pert=None, delta=delta, obs=obs, var=var)
    ds.validate()
    return ds


def test_v3_residual_beats_additive_on_combo():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    ds = _make_residual_dataset()
    split = Split(
        train_idx=np.array([0, 1, 2, 3], dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
        test_idx=np.array([4], dtype=np.int64),
        calib_idx=np.array([], dtype=np.int64),
    )

    additive = AdditiveMeanBaseline()
    additive.fit(ds.delta, ds.obs, split.train_idx)
    add_pred = additive.predict(ds.obs, split.test_idx)
    add_rmse = float(np.sqrt(np.mean((add_pred - ds.delta[split.test_idx]) ** 2)))

    out = fit_predict_perturbfm_v3_residual(
        ds,
        split,
        hidden_dim=16,
        lr=5e-2,
        epochs=200,
        device="cpu",
        use_gating=False,
        gating_mode="none",
        adjacencies=[np.eye(ds.n_genes, dtype=np.float32)],
        seed=0,
    )
    resid_rmse = float(np.sqrt(np.mean((out["mean"] - ds.delta[split.test_idx]) ** 2)))

    assert resid_rmse < add_rmse
