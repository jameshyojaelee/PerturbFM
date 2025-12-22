"""scPerturBench metric panel."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(((y_true - y_pred) ** 2).mean())


def _pcc_delta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _energy_distance(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    return float("nan")


def _wasserstein_distance(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    return float("nan")


def _kl_divergence(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    return float("nan")


def _common_degs(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    return float("nan")


def _aggregate(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> Tuple[Dict[str, float], float]:
    per_group: Dict[str, float] = {}
    weights = []
    vals = []
    for g in np.unique(groups):
        idx = groups == g
        val = metric_fn(y_true[idx], y_pred[idx])
        per_group[str(g)] = val
        if not np.isnan(val):
            weights.append(idx.sum())
            vals.append(val)
    global_weighted = float(np.average(vals, weights=weights)) if weights else float("nan")
    return per_group, global_weighted


def compute_scperturbench_metrics(y_true: np.ndarray, y_pred: np.ndarray, obs: dict) -> Dict[str, object]:
    metrics_fns = {
        "MSE": _mse,
        "PCC_delta": _pcc_delta,
        "Energy": _energy_distance,
        "Wasserstein": _wasserstein_distance,
        "KL": _kl_divergence,
        "Common_DEGs": _common_degs,
    }
    per_pert = {}
    per_context = {}
    global_metrics = {}

    pert_groups = np.asarray(obs["pert_id"])
    ctx_groups = np.asarray(obs["context_id"])
    for name, fn in metrics_fns.items():
        per_pert_vals, global_weighted = _aggregate(fn, y_true, y_pred, pert_groups)
        per_ctx_vals, _ = _aggregate(fn, y_true, y_pred, ctx_groups)
        global_metrics[name] = global_weighted
        for k, v in per_pert_vals.items():
            per_pert.setdefault(k, {})[name] = v
        for k, v in per_ctx_vals.items():
            per_context.setdefault(k, {})[name] = v

    return {
        "global": global_metrics,
        "per_perturbation": per_pert,
        "per_context": per_context,
        "notes": {
            "Energy": "TODO: validate against reference.",
            "Wasserstein": "TODO: validate against reference.",
            "KL": "TODO: validate against reference.",
            "Common_DEGs": "TODO: validate against reference.",
        },
    }
