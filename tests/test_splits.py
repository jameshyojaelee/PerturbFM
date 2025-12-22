import numpy as np
import pytest

from perturbfm.data.splits.split_spec import Split
from perturbfm.data.splits.split_store import SplitStore


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
