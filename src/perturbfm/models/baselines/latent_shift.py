"""Latent shift baseline (linear PCA shift)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class LatentShiftBaseline:
    n_components: int = 32
    mean: np.ndarray | None = None
    components: np.ndarray | None = None
    shifts: Dict[str, np.ndarray] = field(default_factory=dict)

    def fit(self, X_control: np.ndarray, delta: np.ndarray, obs: dict, idx: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        X = X_control[idx]
        self.mean = X.mean(axis=0)
        Xc = X - self.mean
        u, s, vt = np.linalg.svd(Xc, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        self.components = vt[:k]

        delta_proj = delta[idx] @ self.components.T
        pert_ids = np.asarray(obs["pert_id"])[idx]
        for pert in np.unique(pert_ids):
            mask = pert_ids == pert
            self.shifts[str(pert)] = delta_proj[mask].mean(axis=0)

    def predict(self, X_control: np.ndarray, obs: dict, idx: np.ndarray) -> np.ndarray:
        if self.components is None or self.mean is None:
            raise RuntimeError("Model is not fit.")
        idx = np.asarray(idx, dtype=np.int64)
        X = X_control[idx]
        z_control = (X - self.mean) @ self.components.T
        shifts = np.zeros_like(z_control)
        pert_ids = np.asarray(obs["pert_id"])[idx]
        for i, pert in enumerate(pert_ids):
            if str(pert) in self.shifts:
                shifts[i] = self.shifts[str(pert)]
        z_pred = z_control + shifts
        x_pred = z_pred @ self.components + self.mean
        return x_pred - X
