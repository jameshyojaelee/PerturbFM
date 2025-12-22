"""Split specification and generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from perturbfm.utils.hashing import sha256_json


def _canonical_indices(idx: Sequence[int]) -> List[int]:
    return sorted({int(i) for i in idx})


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    ood_axes: Dict[str, List[str]] = field(default_factory=dict)
    seed: int = 0
    frozen_hash: str | None = None
    _is_frozen: bool = False

    def compute_hash(self) -> str:
        payload = {
            "train_idx": _canonical_indices(self.train_idx),
            "val_idx": _canonical_indices(self.val_idx),
            "test_idx": _canonical_indices(self.test_idx),
            "ood_axes": self.ood_axes,
            "seed": self.seed,
        }
        return sha256_json(payload)

    def freeze(self) -> "Split":
        self.frozen_hash = self.compute_hash()
        self._is_frozen = True
        for arr in (self.train_idx, self.val_idx, self.test_idx):
            if isinstance(arr, np.ndarray):
                arr.setflags(write=False)
        return self

    def assert_frozen(self) -> None:
        if not self.frozen_hash:
            raise ValueError("Split is not frozen.")

    def to_dict(self) -> Dict[str, object]:
        return {
            "train_idx": _canonical_indices(self.train_idx),
            "val_idx": _canonical_indices(self.val_idx),
            "test_idx": _canonical_indices(self.test_idx),
            "ood_axes": self.ood_axes,
            "seed": self.seed,
            "frozen_hash": self.frozen_hash,
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "Split":
        split = Split(
            train_idx=np.array(payload["train_idx"], dtype=np.int64),
            val_idx=np.array(payload["val_idx"], dtype=np.int64),
            test_idx=np.array(payload["test_idx"], dtype=np.int64),
            ood_axes=payload.get("ood_axes", {}),
            seed=int(payload.get("seed", 0)),
            frozen_hash=payload.get("frozen_hash"),
        )
        return split


def context_ood_split(
    obs_contexts: Sequence[str],
    holdout_contexts: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Split:
    rng = np.random.default_rng(seed)
    obs_contexts = np.asarray(obs_contexts)
    holdout_contexts = set(holdout_contexts)
    test_idx = np.where(np.isin(obs_contexts, list(holdout_contexts)))[0]
    train_val_idx = np.where(~np.isin(obs_contexts, list(holdout_contexts)))[0]

    rng.shuffle(train_val_idx)
    n_val = max(1, int(len(train_val_idx) * val_fraction)) if len(train_val_idx) > 0 else 0
    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]

    split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ood_axes={"context": sorted(holdout_contexts)},
        seed=seed,
    )
    return split


def leave_one_context_out(
    obs_contexts: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Iterable[Split]:
    contexts = sorted(set(obs_contexts))
    for ctx in contexts:
        yield context_ood_split(obs_contexts, [ctx], seed=seed, val_fraction=val_fraction)


def perturbation_ood_split(
    obs_perts: Sequence[str],
    holdout_perts: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Split:
    rng = np.random.default_rng(seed)
    obs_perts = np.asarray(obs_perts)
    holdout_perts = set(holdout_perts)
    test_idx = np.where(np.isin(obs_perts, list(holdout_perts)))[0]
    train_val_idx = np.where(~np.isin(obs_perts, list(holdout_perts)))[0]
    rng.shuffle(train_val_idx)
    n_val = max(1, int(len(train_val_idx) * val_fraction)) if len(train_val_idx) > 0 else 0
    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]
    split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ood_axes={"perturbation": sorted(holdout_perts)},
        seed=seed,
    )
    return split


def combo_generalization_split(
    obs_perts: Sequence[str],
    holdout_combos: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Split:
    rng = np.random.default_rng(seed)
    obs_perts = np.asarray(obs_perts)
    holdout_combos = set(holdout_combos)
    test_idx = np.where(np.isin(obs_perts, list(holdout_combos)))[0]
    train_val_idx = np.where(~np.isin(obs_perts, list(holdout_combos)))[0]
    rng.shuffle(train_val_idx)
    n_val = max(1, int(len(train_val_idx) * val_fraction)) if len(train_val_idx) > 0 else 0
    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]
    split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ood_axes={"combo": sorted(holdout_combos)},
        seed=seed,
    )
    return split


def covariate_transfer_split(
    covariate: Sequence[str],
    holdout_values: Sequence[str],
    seed: int = 0,
    val_fraction: float = 0.1,
) -> Split:
    rng = np.random.default_rng(seed)
    covariate = np.asarray(covariate)
    holdout_values = set(holdout_values)
    test_idx = np.where(np.isin(covariate, list(holdout_values)))[0]
    train_val_idx = np.where(~np.isin(covariate, list(holdout_values)))[0]
    rng.shuffle(train_val_idx)
    n_val = max(1, int(len(train_val_idx) * val_fraction)) if len(train_val_idx) > 0 else 0
    val_idx = train_val_idx[:n_val]
    train_idx = train_val_idx[n_val:]
    split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        ood_axes={"covariate": sorted(holdout_values)},
        seed=seed,
    )
    return split
