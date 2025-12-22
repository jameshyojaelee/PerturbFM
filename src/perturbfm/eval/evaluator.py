"""Evaluation harness."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics
from perturbfm.eval.uncertainty_metrics import compute_uncertainty_metrics
from perturbfm.eval.report import render_report
from perturbfm.train.trainer import (
    fit_predict_baseline,
    fit_predict_perturbfm_v0,
    fit_predict_perturbfm_v1,
    get_baseline,
)
from perturbfm.utils.hashing import stable_json_dumps


def _default_run_id(split_hash: str, model_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{split_hash[:7]}_{model_name}"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(stable_json_dumps(payload), encoding="utf-8")


def _require_metrics_complete(metrics: Dict[str, object]) -> None:
    required_panels = {
        "scperturbench": ["MSE", "PCC_delta", "Energy", "Wasserstein", "KL", "Common_DEGs"],
        "perturbench": ["RMSE", "RankMetrics", "VarianceDiagnostics"],
        "uncertainty": ["coverage", "nll", "risk_coverage", "ood_auroc"],
    }
    for panel, keys in required_panels.items():
        if panel not in metrics:
            raise ValueError(f"Missing required metrics panel: {panel}")
        panel_obj = metrics[panel]
        if not isinstance(panel_obj, dict):
            raise ValueError(f"Metrics panel {panel} must be a dict.")
        if panel == "uncertainty":
            for k in keys:
                if k not in panel_obj:
                    raise ValueError(f"Missing required uncertainty metric: {k}")
        else:
            global_section = panel_obj.get("global", {})
            for k in keys:
                if k not in global_section:
                    raise ValueError(f"Missing required metric {k} in panel {panel}")


def run_baseline(
    data_path: str,
    split_hash: str,
    baseline_name: str,
    out_dir: Optional[str] = None,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    model = get_baseline(baseline_name, **kwargs)
    preds = fit_predict_baseline(model, ds, split)

    run_id = _default_run_id(split_hash, baseline_name)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "split_hash": split_hash,
        "model": {"name": baseline_name, **kwargs},
    }
    _write_json(run_dir / "config.json", config)
    (run_dir / "split_hash.txt").write_text(split_hash + "\n", encoding="utf-8")

    report_html = render_report(metrics)
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")

    return run_dir


def run_perturbfm_v0(
    data_path: str,
    split_hash: str,
    out_dir: Optional[str] = None,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    preds = fit_predict_perturbfm_v0(ds, split, **kwargs)
    run_id = _default_run_id(split_hash, "perturbfm_v0")
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {"data_path": data_path, "split_hash": split_hash, "model": {"name": "perturbfm_v0", **kwargs}}
    _write_json(run_dir / "config.json", config)
    (run_dir / "split_hash.txt").write_text(split_hash + "\n", encoding="utf-8")

    report_html = render_report(metrics)
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    return run_dir


def run_perturbfm_v1(
    data_path: str,
    split_hash: str,
    adjacency,
    pert_gene_masks,
    out_dir: Optional[str] = None,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    preds = fit_predict_perturbfm_v1(ds, split, adjacency=adjacency, pert_gene_masks=pert_gene_masks, **kwargs)
    run_id = _default_run_id(split_hash, "perturbfm_v1")
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {"data_path": data_path, "split_hash": split_hash, "model": {"name": "perturbfm_v1", **kwargs}}
    _write_json(run_dir / "config.json", config)
    (run_dir / "split_hash.txt").write_text(split_hash + "\n", encoding="utf-8")

    report_html = render_report(metrics)
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    return run_dir


def evaluate_predictions(
    data_path: str,
    split_hash: str,
    predictions_path: str,
    out_dir: str,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    _ = store.load(split_hash)

    npz = np.load(predictions_path)
    mean = npz["mean"]
    var = npz["var"]
    idx = npz["idx"]
    subset = ds.select(idx)
    y_true = subset.delta

    metrics_sc = compute_scperturbench_metrics(y_true, mean, subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, mean, subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, mean, var, ood_labels=ood_labels)

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _write_json(out_path / "metrics.json", metrics)
    _write_json(out_path / "calibration.json", metrics_unc)
    report_html = render_report(metrics)
    (out_path / "report.html").write_text(report_html, encoding="utf-8")
    return out_path
