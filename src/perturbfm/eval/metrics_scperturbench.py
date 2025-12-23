"""scPerturBench metric panel."""

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


def _energy_distance(_y_true: np.ndarray, _y_pred: np.ndarray, max_n: int = 200, seed: int = 0) -> float:
    """
    Energy distance between two multivariate samples.
    Scalable default: subsample to at most max_n points per set.
    """
    rng = np.random.default_rng(seed)
    x = _y_true.reshape(_y_true.shape[0], -1)
    y = _y_pred.reshape(_y_pred.shape[0], -1)
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]
    dx = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2))
    dy = np.sqrt(((y[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    dxy = np.sqrt(((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2))
    term = 2.0 * dxy.mean() - dx.mean() - dy.mean()
    return float(term)


def _wasserstein_distance(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    """
    Approximate per-gene 1D Wasserstein by fixed quantile matching and average across genes.
    This is a scalable proxy for multivariate Wasserstein without external deps.
    """
    x = _y_true
    y = _y_pred
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    qs = np.linspace(0, 100, num=51)
    dists = []
    for g in range(x.shape[1]):
        qx = np.percentile(x[:, g], qs)
        qy = np.percentile(y[:, g], qs)
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
    Common-DEGs: overlap of top-k genes ranked by |mean delta|.
    Deterministic and scale-friendly.
    """
    k = min(k, _y_true.shape[1])
    true_rank = np.argsort(-np.abs(_y_true).mean(axis=0))[:k]
    pred_rank = np.argsort(-np.abs(_y_pred).mean(axis=0))[:k]
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
            "Energy": "Subsampled energy distance (max_n=200).",
            "Wasserstein": "Per-gene quantile matching approximation.",
            "KL": "Diagonal Gaussian assumption.",
            "Common_DEGs": "Top-k overlap by |mean delta|.",
        },
    }
