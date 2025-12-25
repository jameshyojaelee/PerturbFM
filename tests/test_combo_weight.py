import numpy as np
import pytest

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import Split
from perturbfm.train.trainer import fit_predict_perturbfm_v2_residual


def _make_combo_heavy_dataset(n_single: int = 20) -> PerturbDataset:
    delta = []
    pert_genes = []
    pert_id = []
    for _ in range(n_single):
        delta.append([1.0, 0.0, 0.0])
        pert_genes.append(["A"])
        pert_id.append("A")
    for _ in range(n_single):
        delta.append([0.0, 1.0, 0.0])
        pert_genes.append(["B"])
        pert_id.append("B")
    for _ in range(n_single):
        delta.append([0.0, 0.0, 1.0])
        pert_genes.append(["C"])
        pert_id.append("C")
    # combo row (train) and identical combo row (test)
    delta.append([3.0, 3.0, 2.0])
    pert_genes.append(["A", "B"])
    pert_id.append("A+B")
    delta.append([3.0, 3.0, 2.0])
    pert_genes.append(["A", "B"])
    pert_id.append("A+B")

    delta = np.array(delta, dtype=np.float32)
    obs = {
        "pert_id": pert_id,
        "context_id": ["C0"] * len(delta),
        "batch_id": ["B0"] * len(delta),
        "is_control": [False] * len(delta),
        "pert_genes": pert_genes,
    }
    var = ["A", "B", "C"]
    ds = PerturbDataset(X_control=None, X_pert=None, delta=delta, obs=obs, var=var)
    ds.validate()
    return ds


def test_combo_weight_improves_combo_rmse():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    ds = _make_combo_heavy_dataset()
    train_idx = np.arange(len(ds.delta) - 1, dtype=np.int64)
    split = Split(
        train_idx=train_idx,
        val_idx=np.array([], dtype=np.int64),
        test_idx=np.array([len(ds.delta) - 1], dtype=np.int64),
        calib_idx=np.array([], dtype=np.int64),
    )

    def _run(combo_weight: float) -> float:
        torch.manual_seed(0)
        out = fit_predict_perturbfm_v2_residual(
            ds,
            split,
            hidden_dim=16,
            lr=1e-2,
            epochs=50,
            device="cpu",
            use_gating=False,
            gating_mode="none",
            contextual_operator=True,
            num_bases=2,
            adjacencies=[np.eye(ds.n_genes, dtype=np.float32)],
            seed=0,
            combo_weight=combo_weight,
        )
        return float(np.sqrt(np.mean((out["mean"] - ds.delta[split.test_idx]) ** 2)))

    rmse_unweighted = _run(1.0)
    rmse_weighted = _run(4.0)
    assert rmse_weighted < rmse_unweighted
