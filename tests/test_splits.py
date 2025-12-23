import numpy as np
import pytest

from perturbfm.data.splits.split_spec import Split
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.data.splits import split_spec


def test_split_hash_order_invariant():
    a = Split(train_idx=np.array([2, 1, 0]), val_idx=np.array([3]), test_idx=np.array([4]), ood_axes={}, seed=0)
    b = Split(train_idx=np.array([0, 1, 2]), val_idx=np.array([3]), test_idx=np.array([4]), ood_axes={}, seed=0)
    assert a.compute_hash() == b.compute_hash()


def test_split_hash_changes_on_index_modification():
    a = Split(train_idx=np.array([0, 1]), val_idx=np.array([2]), test_idx=np.array([3]), ood_axes={}, seed=0)
    b = Split(train_idx=np.array([0, 2]), val_idx=np.array([2]), test_idx=np.array([3]), ood_axes={}, seed=0)
    assert a.compute_hash() != b.compute_hash()


def test_split_store_refuses_overwrite(tmp_path):
    store = SplitStore(root=tmp_path)
    split = Split(train_idx=np.array([0]), val_idx=np.array([1]), test_idx=np.array([2]), ood_axes={}, seed=0).freeze()
    store.save(split)
    bad = Split(train_idx=np.array([1]), val_idx=np.array([2]), test_idx=np.array([3]), ood_axes={}, seed=0)
    bad.frozen_hash = split.frozen_hash
    with pytest.raises(ValueError):
        store.save(bad)


def test_freeze_prevents_mutation():
    split = Split(train_idx=np.array([0, 1]), val_idx=np.array([2]), test_idx=np.array([3]), ood_axes={}, seed=0).freeze()
    with pytest.raises(ValueError):
        split.train_idx[0] = 99


def test_perturbation_ood_split_disjoint():
    perts = ["A", "B", "C", "A", "B"]
    split = split_spec.perturbation_ood_split(perts, holdout_perts=["C"])
    assert set(split.test_idx.tolist()) == {2}
    assert set(split.train_idx).isdisjoint(split.test_idx)


def test_covariate_transfer_split_disjoint():
    covs = ["low", "high", "low", "mid"]
    split = split_spec.covariate_transfer_split(covs, holdout_values=["high"])
    assert set(split.test_idx.tolist()) == {1}


def test_context_ood_filters_unshared_perts():
    contexts = ["C0", "C0", "C1", "C1"]
    perts = ["P0", "P1", "P2", "P2"]
    split = split_spec.context_ood_split(contexts, ["C1"], obs_perts=perts, seed=0, val_fraction=0.5)
    # P2 not in train; test should be filtered empty
    assert len(split.test_idx) == 0
    assert "context" in split.ood_axes
    assert "perturbation" in split.ood_axes
    assert split.notes.get("warning") == "test_filtered_for_shared_perturbations"
