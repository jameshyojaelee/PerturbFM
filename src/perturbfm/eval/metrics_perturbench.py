"""PerturBench metric panel."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    ar = np.argsort(np.argsort(a)).astype(np.float64)
    br = np.argsort(np.argsort(b)).astype(np.float64)
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = np.sqrt((ar**2).sum()) * np.sqrt((br**2).sum())
    if denom == 0:
        return float("nan")
    return float((ar * br).sum() / denom)


def _rank_metrics(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    """
    Rank-based metric: Spearman correlation of mean delta gene ranks per group.
    This aligns with perturbation-level DEG ranking fidelity.
    """
    if _y_true.shape[0] == 0:
        return float("nan")
    true_mean = _y_true.mean(axis=0)
    pred_mean = _y_pred.mean(axis=0)
    return _spearman_corr(true_mean, pred_mean)


def _topk_overlap(_y_true: np.ndarray, _y_pred: np.ndarray, k: int = 50) -> float:
    """
    Top-k overlap of genes ranked by |mean delta| in true vs predicted.
    """
    k = min(k, _y_true.shape[1])
    if k == 0:
        return float("nan")
    true_rank = np.argsort(-np.abs(_y_true.mean(axis=0)))[:k]
    pred_rank = np.argsort(-np.abs(_y_pred.mean(axis=0)))[:k]
    overlap = len(set(true_rank.tolist()) & set(pred_rank.tolist()))
    return float(overlap / k)


def _variance_diagnostics(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    """
    Collapse diagnostic: ratio of predicted variance to true variance.
    Values near 1 are healthy; values near 0 indicate collapse.
    """
    true_var = _y_true.var(axis=0).mean()
    pred_var = _y_pred.var(axis=0).mean()
    return float(pred_var / (true_var + 1e-8))


def _mean_l2_distance(a: np.ndarray, max_pairs: int, rng: np.random.Generator) -> float:
    if a.shape[0] == 0:
        return float("nan")
    total_pairs = a.shape[0] * a.shape[0]
    if total_pairs <= max_pairs:
        diffs = a[:, None, :] - a[None, :, :]
        return float(np.sqrt((diffs**2).sum(axis=2)).mean())
    idx_a = rng.integers(0, a.shape[0], size=max_pairs)
    idx_b = rng.integers(0, a.shape[0], size=max_pairs)
    diffs = a[idx_a] - a[idx_b]
    return float(np.sqrt((diffs**2).sum(axis=1)).mean())


def _diversity_ratio(_y_true: np.ndarray, _y_pred: np.ndarray, max_pairs: int = 20000, seed: int = 0) -> float:
    """
    Collapse diagnostic: ratio of mean pairwise distances (pred/true).
    """
    rng = np.random.default_rng(seed)
    true_div = _mean_l2_distance(_y_true, max_pairs, rng)
    pred_div = _mean_l2_distance(_y_pred, max_pairs, rng)
    return float(pred_div / (true_div + 1e-8))


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
        "TopKOverlap": _topk_overlap,
        "VarianceDiagnostics": _variance_diagnostics,
        "DiversityRatio": _diversity_ratio,
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
            "RankMetrics": "Spearman correlation of mean delta gene ranks per group.",
            "TopKOverlap": "Top-k overlap of |mean delta| genes (k=50).",
            "VarianceDiagnostics": "Predicted variance / true variance (collapse diagnostic).",
            "DiversityRatio": "Pairwise distance ratio (pred/true) with sampling.",
        },
    }
