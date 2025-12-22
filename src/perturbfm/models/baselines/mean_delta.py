"""Mean-delta baselines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class MeanDeltaBaseline:
    mode: str = "global"
    global_mean: np.ndarray | None = None
    group_means: Dict[Tuple[str, str] | str, np.ndarray] = field(default_factory=dict)

    def fit(self, delta: np.ndarray, obs: dict, idx: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        train_delta = delta[idx]
        self.global_mean = train_delta.mean(axis=0)
        if self.mode == "global":
            return

        pert_ids = np.asarray(obs["pert_id"])[idx]
        if self.mode == "per_perturbation":
            for pert in np.unique(pert_ids):
                mask = pert_ids == pert
                self.group_means[str(pert)] = train_delta[mask].mean(axis=0)
            return

        if self.mode == "per_perturbation_context":
            contexts = np.asarray(obs["context_id"])[idx]
            for pert in np.unique(pert_ids):
                for ctx in np.unique(contexts):
                    mask = (pert_ids == pert) & (contexts == ctx)
                    if mask.any():
                        self.group_means[(str(pert), str(ctx))] = train_delta[mask].mean(axis=0)
            return

        raise ValueError(f"Unknown mean baseline mode: {self.mode}")

    def predict(self, obs: dict, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx, dtype=np.int64)
        if self.global_mean is None:
            raise RuntimeError("Model is not fit.")
        n = len(idx)
        out = np.tile(self.global_mean, (n, 1))

        if self.mode == "global":
            return out

        pert_ids = np.asarray(obs["pert_id"])[idx]
        if self.mode == "per_perturbation":
            for i, pert in enumerate(pert_ids):
                if str(pert) in self.group_means:
                    out[i] = self.group_means[str(pert)]
            return out

        if self.mode == "per_perturbation_context":
            contexts = np.asarray(obs["context_id"])[idx]
            for i, (pert, ctx) in enumerate(zip(pert_ids, contexts)):
                key = (str(pert), str(ctx))
                if key in self.group_means:
                    out[i] = self.group_means[key]
            return out

        raise ValueError(f"Unknown mean baseline mode: {self.mode}")
