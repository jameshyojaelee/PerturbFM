"""Diagnostics and validation hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics


def validate_metrics(predictions_path: str, data_path: str, external_script: Optional[str] = None) -> dict:
    ds = PerturbDataset.load_artifact(data_path)
    npz = np.load(predictions_path)
    idx = npz["idx"]
    mean = npz["mean"]
    y_true = ds.delta[idx]

    metrics = {
        "scperturbench": compute_scperturbench_metrics(y_true, mean, ds.obs),
        "perturbench": compute_perturbench_metrics(y_true, mean, ds.obs),
    }

    if external_script:
        raise NotImplementedError(
            "External validation hooks should execute scripts under third_party/ without vendoring GPL code."
        )
    return metrics
