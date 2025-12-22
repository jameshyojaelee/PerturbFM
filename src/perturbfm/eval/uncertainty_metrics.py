"""Uncertainty and calibration metrics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def _gaussian_nll(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
    var = np.clip(var, 1e-6, None)
    return float(0.5 * (np.log(var) + (y_true - mean) ** 2 / var).mean())


def _coverage(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, nominal: float) -> float:
    std = np.sqrt(np.clip(var, 1e-6, None))
    z = _z_from_nominal(nominal)
    lower = mean - z * std
    upper = mean + z * std
    within = (y_true >= lower) & (y_true <= upper)
    return float(within.mean())


def _z_from_nominal(nominal: float) -> float:
    # Approximate inverse CDF for common nominals
    lookup = {0.5: 0.67449, 0.8: 1.28155, 0.9: 1.64485, 0.95: 1.95996}
    if nominal in lookup:
        return lookup[nominal]
    return 1.0


def _risk_coverage_curve(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> Dict[str, List[float]]:
    n = y_true.shape[0]
    uncertainty = var.mean(axis=1)
    order = np.argsort(uncertainty)
    risks = []
    coverages = []
    for k in range(1, n + 1):
        idx = order[:k]
        mse = ((y_true[idx] - mean[idx]) ** 2).mean()
        risks.append(float(mse))
        coverages.append(float(k / n))
    return {"coverage": coverages, "risk": risks}


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    labels = labels.astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1
    n_pos = pos.sum()
    n_neg = len(labels) - n_pos
    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_uncertainty_metrics(
    y_true: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    ood_labels: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    nominals = [0.5, 0.8, 0.9, 0.95]
    coverage = {str(n): _coverage(y_true, mean, var, n) for n in nominals}
    risk_curve = _risk_coverage_curve(y_true, mean, var)
    ood_auc = None
    if ood_labels is not None:
        scores = var.mean(axis=1)
        ood_auc = _auroc(scores, np.asarray(ood_labels))
    return {
        "coverage": coverage,
        "nll": _gaussian_nll(y_true, mean, var),
        "risk_coverage": risk_curve,
        "ood_auroc": ood_auc,
        "notes": {
            "ood_auroc": "Requires ood_labels; uses mean variance per sample as score.",
        },
    }
