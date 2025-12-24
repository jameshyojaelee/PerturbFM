import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import Split
from perturbfm.models.baselines.additive_mean import AdditiveMeanBaseline
from perturbfm.train.trainer import get_baseline


def _make_toy_dataset() -> PerturbDataset:
    delta = np.array(
        [
            [1.0, 2.0, 0.0],  # single A (train)
            [-1.0, 0.0, 3.0],  # single B (train)
            [9.0, 9.0, 9.0],  # double A+B (test)
            [100.0, 100.0, 100.0],  # single A (test, should not affect fit)
            [5.0, 5.0, 5.0],  # double A+C (test, C unseen)
            [0.0, 0.0, 0.0],  # control
        ],
        dtype=np.float32,
    )
    obs = {
        "pert_id": ["A", "B", "A+B", "A", "A+C", "control"],
        "context_id": ["C0"] * 6,
        "batch_id": ["B0"] * 6,
        "is_control": [False, False, False, False, False, True],
        "pert_genes": [["A"], ["B"], ["A", "B"], ["A"], ["A", "C"], []],
    }
    var = ["A", "B", "C"]
    ds = PerturbDataset(X_control=None, X_pert=None, delta=delta, obs=obs, var=var)
    ds.validate()
    return ds


def test_additive_mean_baseline_math_and_missing():
    ds = _make_toy_dataset()
    split = Split(
        train_idx=np.array([0, 1], dtype=np.int64),
        val_idx=np.array([], dtype=np.int64),
        test_idx=np.array([2, 3, 4, 5], dtype=np.int64),
        calib_idx=np.array([], dtype=np.int64),
    )

    model = AdditiveMeanBaseline()
    model.fit(ds.delta, ds.obs, split.train_idx)

    assert np.allclose(model.gene_means["A"], ds.delta[0])
    assert np.allclose(model.gene_means["B"], ds.delta[1])

    preds = model.predict(ds.obs, split.test_idx)
    expected_ab = ds.delta[0] + ds.delta[1]
    assert np.allclose(preds[0], expected_ab)
    assert np.allclose(preds[1], ds.delta[0])
    assert np.allclose(preds[2], ds.delta[0])
    assert np.allclose(preds[3], np.zeros(ds.n_genes, dtype=ds.delta.dtype))
    assert model.missing_contribs == 1
    assert model.missing_genes == ["C"]


def test_additive_mean_baseline_registered():
    model = get_baseline("additive_mean")
    assert isinstance(model, AdditiveMeanBaseline)
