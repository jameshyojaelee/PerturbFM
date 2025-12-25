"""Common data transforms."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, List

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


def pert_genes_to_mask(pert_genes: Sequence[List[str]], var: Sequence[str]) -> np.ndarray:
    gene_to_idx = {g: i for i, g in enumerate(var)}
    mask = np.zeros((len(pert_genes), len(var)), dtype=np.float32)
    for i, genes in enumerate(pert_genes):
        for g in genes:
            if g in gene_to_idx:
                mask[i, gene_to_idx[g]] = 1.0
    return mask


def select_hvg_train_only(delta: np.ndarray, train_idx: Sequence[int], n_hvg: int) -> np.ndarray:
    if delta is None:
        raise ValueError("delta required for HVG selection.")
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if train_idx.size == 0:
        raise ValueError("train_idx is empty for HVG selection.")
    train_delta = delta[train_idx]
    var = np.var(train_delta, axis=0)
    if n_hvg >= var.shape[0]:
        return np.arange(var.shape[0], dtype=np.int64)
    top = np.argpartition(var, -n_hvg)[-n_hvg:]
    return np.sort(top)
