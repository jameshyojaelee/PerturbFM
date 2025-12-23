"""Split specification and generators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from perturbfm.utils.hashing import sha256_json


def _canonical_indices(idx: Sequence[int]) -> List[int]:
    return sorted({int(i) for i in idx})


def _derive_calib_idx(val_idx: Sequence[int], seed: int, split_hash: str | None = None, frac: float = 0.5) -> np.ndarray:
    val_idx = np.array(list(val_idx), dtype=np.int64)
    if val_idx.size == 0:
        return np.array([], dtype=np.int64)
    token = split_hash or sha256_json(_canonical_indices(val_idx))
    seed_mix = seed ^ int(token[:8], 16)
    rng = np.random.default_rng(seed_mix)
    idx = val_idx.copy()
    rng.shuffle(idx)
    n_calib = max(1, int(round(len(idx) * frac)))
    n_calib = min(n_calib, len(idx))
    return idx[:n_calib]


@dataclass
class Split:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    calib_idx: np.ndarray | None = None
    ood_axes: Dict[str, List[str]] = field(default_factory=dict)
    notes: Dict[str, object] = field(default_factory=dict)
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
        if self.calib_idx is not None:
            payload["calib_idx"] = _canonical_indices(self.calib_idx)
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
            "calib_idx": _canonical_indices(self.calib_idx) if self.calib_idx is not None else None,
            "ood_axes": self.ood_axes,
            "notes": self.notes,
            "seed": self.seed,
            "frozen_hash": self.frozen_hash,
        }

    @staticmethod
    def from_dict(payload: Dict[str, object]) -> "Split":
        split = Split(
            train_idx=np.array(payload["train_idx"], dtype=np.int64),
            val_idx=np.array(payload["val_idx"], dtype=np.int64),
            test_idx=np.array(payload["test_idx"], dtype=np.int64),
            calib_idx=np.array(payload["calib_idx"], dtype=np.int64) if payload.get("calib_idx") is not None else None,
            ood_axes=payload.get("ood_axes", {}),
            notes=payload.get("notes", {}),
            seed=int(payload.get("seed", 0)),
            frozen_hash=payload.get("frozen_hash"),
        )
        return split


def context_ood_split(
    obs_contexts: Sequence[str],
    holdout_contexts: Sequence[str],
    obs_perts: Sequence[str] | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    require_shared_perturbations: bool = True,
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

    notes = {}
    ood_axes = {"context": sorted(holdout_contexts)}

    if require_shared_perturbations:
        if obs_perts is None:
            raise ValueError("obs_perts required when require_shared_perturbations=True")
        obs_perts = np.asarray(obs_perts)
        train_perts = set(obs_perts[train_idx].tolist())
        test_perts = set(obs_perts[test_idx].tolist())
        missing = sorted(test_perts - train_perts)
        if missing:
            # Filter test to shared perturbations (Option A)
            shared_mask = np.isin(obs_perts[test_idx], list(train_perts))
            filtered_test_idx = test_idx[shared_mask]
            notes["warning"] = "test_filtered_for_shared_perturbations"
            notes["dropped_test_count"] = int(len(test_idx) - len(filtered_test_idx))
            notes["missing_test_perts"] = missing
            test_idx = filtered_test_idx
            ood_axes["perturbation"] = ["mixed_ood_filtered"]

    split = Split(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        calib_idx=_derive_calib_idx(val_idx, seed),
        ood_axes=ood_axes,
        notes=notes,
        seed=seed,
    )
    return split


def leave_one_context_out(
    obs_contexts: Sequence[str],
    obs_perts: Sequence[str] | None = None,
    seed: int = 0,
    val_fraction: float = 0.1,
    require_shared_perturbations: bool = True,
) -> Iterable[Split]:
    contexts = sorted(set(obs_contexts))
    for ctx in contexts:
        yield context_ood_split(
            obs_contexts,
            [ctx],
            obs_perts=obs_perts,
            seed=seed,
            val_fraction=val_fraction,
            require_shared_perturbations=require_shared_perturbations,
        )


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
        calib_idx=_derive_calib_idx(val_idx, seed),
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
        calib_idx=_derive_calib_idx(val_idx, seed),
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
        calib_idx=_derive_calib_idx(val_idx, seed),
        ood_axes={"covariate": sorted(holdout_values)},
        seed=seed,
    )
    return split
