"""scPerturBench metric panel with scalable defaults."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error across all genes and samples."""
    return float(((y_true - y_pred) ** 2).mean())


def _pcc_delta(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation over flattened delta vectors."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    if y_true.std() == 0 or y_pred.std() == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])

def _mean_l2_distance(
    a: np.ndarray, b: np.ndarray, max_pairs: int, rng: np.random.Generator
) -> float:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    total_pairs = a.shape[0] * b.shape[0]
    if total_pairs <= max_pairs:
        diffs = a[:, None, :] - b[None, :, :]
        return float(np.sqrt((diffs**2).sum(axis=2)).mean())
    idx_a = rng.integers(0, a.shape[0], size=max_pairs)
    idx_b = rng.integers(0, b.shape[0], size=max_pairs)
    diffs = a[idx_a] - b[idx_b]
    return float(np.sqrt((diffs**2).sum(axis=1)).mean())


def _energy_distance(_y_true: np.ndarray, _y_pred: np.ndarray, max_pairs: int = 20000, seed: int = 0) -> float:
    """
    Energy distance between two multivariate samples.

    Computed as 2 E||X-Y|| - E||X-X'|| - E||Y-Y'|| with pair sampling to avoid O(n^2).
    """
    rng = np.random.default_rng(seed)
    x = _y_true.reshape(_y_true.shape[0], -1)
    y = _y_pred.reshape(_y_pred.shape[0], -1)
    dxy = _mean_l2_distance(x, y, max_pairs, rng)
    dxx = _mean_l2_distance(x, x, max_pairs, rng)
    dyy = _mean_l2_distance(y, y, max_pairs, rng)
    return float(2.0 * dxy - dxx - dyy)


def _wasserstein_distance(
    _y_true: np.ndarray,
    _y_pred: np.ndarray,
    n_projections: int = 16,
    n_quantiles: int = 101,
    seed: int = 0,
) -> float:
    """
    Sliced Wasserstein approximation using random projections and quantile matching.
    """
    x = _y_true.reshape(_y_true.shape[0], -1)
    y = _y_pred.reshape(_y_pred.shape[0], -1)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    qs = np.linspace(0, 100, num=n_quantiles)
    dists = []
    for _ in range(n_projections):
        v = rng.normal(size=x.shape[1])
        v /= np.linalg.norm(v) + 1e-8
        px = x @ v
        py = y @ v
        qx = np.percentile(px, qs)
        qy = np.percentile(py, qs)
        dists.append(np.mean(np.abs(qx - qy)))
    return float(np.mean(dists))


def _kl_divergence(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    """
    KL divergence between diagonal Gaussians fitted to true/pred distributions.
    Assumes gene-wise independence.
    """
    mu_p = _y_true.mean(axis=0)
    mu_q = _y_pred.mean(axis=0)
    var_p = _y_true.var(axis=0) + 1e-8
    var_q = _y_pred.var(axis=0) + 1e-8
    kl = 0.5 * np.sum(np.log(var_q / var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1)
    return float(kl / _y_true.shape[1])

def _common_degs(_y_true: np.ndarray, _y_pred: np.ndarray, k: int = 100) -> float:
    """
    Common-DEGs: overlap of top-k genes ranked by |effect size| (mean / std).
    Deterministic and scale-friendly.
    """
    k = min(k, _y_true.shape[1])
    true_score = np.abs(_y_true.mean(axis=0)) / (_y_true.std(axis=0) + 1e-8)
    pred_score = np.abs(_y_pred.mean(axis=0)) / (_y_pred.std(axis=0) + 1e-8)
    true_rank = np.argsort(-true_score)[:k]
    pred_rank = np.argsort(-pred_score)[:k]
    overlap = len(set(true_rank.tolist()) & set(pred_rank.tolist()))
    return float(overlap / k) if k > 0 else float("nan")


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
    """
    Compute scPerturBench-style metrics with scalable approximations.

    Metrics include MSE, PCC_delta, Energy (sampled), Wasserstein (sliced), KL (diag Gaussian),
    and Common_DEGs (top-k overlap by effect size). Aggregates are reported globally and per group.
    """
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
            "Energy": "Energy distance with pair sampling (max_pairs=20000).",
            "Wasserstein": "Sliced Wasserstein via random projections + quantile matching.",
            "KL": "Diagonal Gaussian assumption (gene-wise independence).",
            "Common_DEGs": "Top-k overlap by |mean/std| effect size.",
        },
    }
