"""Mini-batch utilities for PerturbDataset."""

from __future__ import annotations

from typing import Dict, Iterator, Sequence

import numpy as np

from perturbfm.data.canonical import PerturbDataset


def iter_index_batches(
    idx: Sequence[int],
    batch_size: int | None,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[np.ndarray]:
    idx = np.asarray(idx, dtype=np.int64)
    if batch_size is None or batch_size <= 0 or batch_size >= len(idx):
        yield idx
        return
    rng = np.random.default_rng(seed)
    order = idx.copy()
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def batch_iterator(
    dataset: PerturbDataset,
    idx: Sequence[int],
    batch_size: int | None,
    seed: int = 0,
    shuffle: bool = True,
) -> Iterator[Dict[str, object]]:
    """
    Yield mini-batches with common fields (x_control, delta, pert_id, context_id).
    """
    for batch_idx in iter_index_batches(idx, batch_size=batch_size, seed=seed, shuffle=shuffle):
        batch: Dict[str, object] = {"idx": batch_idx}
        if dataset.X_control is not None:
            batch["x_control"] = dataset.X_control[batch_idx]
        if dataset.delta is not None:
            batch["delta"] = dataset.delta[batch_idx]
        if "pert_id" in dataset.obs:
            batch["pert_id"] = [dataset.obs["pert_id"][i] for i in batch_idx]
        if "context_id" in dataset.obs:
            batch["context_id"] = [dataset.obs["context_id"][i] for i in batch_idx]
        if "pert_genes" in dataset.obs:
            batch["pert_genes"] = [dataset.obs["pert_genes"][i] for i in batch_idx]
        yield batch
