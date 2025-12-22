"""PerturBench metric panel."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def _rank_metrics(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    # Spearman rank correlation averaged over samples.
    y_true = _y_true
    y_pred = _y_pred
    if y_true.shape[0] == 0:
        return float("nan")
    corrs = []
    for i in range(y_true.shape[0]):
        t = y_true[i]
        p = y_pred[i]
        # rank transform
        tr = np.argsort(np.argsort(t))
        pr = np.argsort(np.argsort(p))
        cov = np.cov(tr, pr, bias=True)[0, 1]
        std = tr.std() * pr.std()
        if std == 0:
            continue
        corrs.append(cov / std)
    return float(np.mean(corrs)) if corrs else float("nan")


def _variance_diagnostics(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    mse = (( _y_true - _y_pred) ** 2).mean()
    pred_var = _y_pred.var()
    return float(pred_var - mse)


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


def compute_perturbench_metrics(y_true: np.ndarray, y_pred: np.ndarray, obs: dict) -> Dict[str, object]:
    metrics_fns = {
        "RMSE": _rmse,
        "RankMetrics": _rank_metrics,
        "VarianceDiagnostics": _variance_diagnostics,
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
            "RankMetrics": "TODO: define rank-based metrics.",
            "VarianceDiagnostics": "TODO: define prediction variance diagnostics.",
        },
    }
