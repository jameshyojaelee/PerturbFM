"""Simple split conformal prediction for per-gene regression."""

from __future__ import annotations

import numpy as np


def conformal_intervals(residuals: np.ndarray, alphas: list[float]) -> dict:
    """
    residuals: absolute residuals array [N, G]
    alphas: list of coverage levels, e.g., [0.1, 0.05] for 90/95% intervals
    Returns dict mapping coverage string -> quantile value per gene.
    """
    qs = {}
    for a in alphas:
        q = np.quantile(residuals, 1 - a, axis=0)
        qs[str(1 - a)] = q
    return qs
