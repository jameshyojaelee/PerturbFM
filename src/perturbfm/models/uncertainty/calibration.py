"""Calibration utilities."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(confidence: np.ndarray, correctness: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        if not mask.any():
            continue
        acc = correctness[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)
