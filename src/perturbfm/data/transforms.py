"""Common data transforms."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np


def compute_delta(X_control: np.ndarray, X_pert: np.ndarray) -> np.ndarray:
    return X_pert - X_control


def standardize(X: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0, keepdims=True)
    if std is None:
        std = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mean) / std
    return Xn, mean, std


def match_controls_by_context(obs: Dict[str, Sequence], X_control: np.ndarray) -> np.ndarray:
    contexts = np.array(obs["context_id"])
    matched = np.zeros_like(X_control)
    for ctx in np.unique(contexts):
        idx = np.where(contexts == ctx)[0]
        matched[idx] = X_control[idx].mean(axis=0, keepdims=True)
    return matched
