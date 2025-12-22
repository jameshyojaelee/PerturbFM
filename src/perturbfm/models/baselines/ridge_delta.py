"""Ridge regression baseline to predict delta from control expression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RidgeDeltaBaseline:
    alpha: float = 1.0
    weights: np.ndarray | None = None  # shape (G+1, G)

    def fit(self, X_control: np.ndarray, delta: np.ndarray, idx: np.ndarray) -> None:
        idx = np.asarray(idx, dtype=np.int64)
        X = X_control[idx]
        Y = delta[idx]
        n, g = X.shape
        X_aug = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
        reg = self.alpha * np.eye(g + 1, dtype=X.dtype)
        self.weights = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)

    def predict(self, X_control: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fit.")
        idx = np.asarray(idx, dtype=np.int64)
        X = X_control[idx]
        n = X.shape[0]
        X_aug = np.concatenate([X, np.ones((n, 1), dtype=X.dtype)], axis=1)
        return X_aug @ self.weights
