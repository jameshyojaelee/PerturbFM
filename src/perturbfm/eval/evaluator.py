"""Evaluation harness."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.data.splits.split_spec import _derive_calib_idx
from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics
from perturbfm.eval.uncertainty_metrics import compute_uncertainty_metrics
from perturbfm.eval.report import render_report
from perturbfm.models.uncertainty.conformal import conformal_intervals
from perturbfm.train.trainer import (
    _predict_baseline,
    fit_predict_baseline,
    fit_predict_perturbfm_v0,
    fit_predict_perturbfm_v1,
    fit_predict_perturbfm_v2,
    fit_predict_perturbfm_v3,
    get_baseline,
)
from perturbfm.data.transforms import pert_genes_to_mask
from perturbfm.utils.hashing import stable_json_dumps, sha256_json
from perturbfm.utils.config import config_hash


def _default_run_id(split_hash: str, model_name: str, cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}_{split_hash[:7]}_{model_name}_{cfg_hash}"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(stable_json_dumps(payload), encoding="utf-8")


def _dataset_hash(data_path: Path) -> str:
    meta = (data_path / "meta.json").read_text(encoding="utf-8")
    return sha256_json(meta)[:8]


def _ensemble_predictions(run_fn, ensemble_size: int):
    preds_list = []
    for _ in range(ensemble_size):
        preds_list.append(run_fn())
    mean_stack = np.stack([p["mean"] for p in preds_list], axis=0)
    var_stack = np.stack([p["var"] for p in preds_list], axis=0)
    mean = mean_stack.mean(axis=0)
    aleatoric = var_stack.mean(axis=0)
    epistemic = mean_stack.var(axis=0)
    total_var = aleatoric + epistemic
    models = [p.get("model") for p in preds_list if p.get("model") is not None]
    meta = {k: v for k, v in preds_list[0].items() if k not in ("mean", "var", "idx", "model")}
    out = {"mean": mean, "var": total_var, "idx": preds_list[0]["idx"]}
    if models:
        out["models"] = models
    out.update(meta)
    return out


def _get_calib_idx(split) -> tuple[np.ndarray, str]:
    if split.calib_idx is not None:
        return np.asarray(split.calib_idx, dtype=np.int64), "split.calib_idx"
    calib = _derive_calib_idx(split.val_idx, split.seed, split.frozen_hash or split.compute_hash())
    return calib, "derived_from_val"


def _assert_no_test_leak(calib_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if np.intersect1d(calib_idx, test_idx).size > 0:
        raise ValueError("Calibration indices overlap test indices (leakage).")


def _predict_baseline_models(models, dataset: PerturbDataset, idx: np.ndarray) -> np.ndarray:
    preds = []
    for model in models:
        preds.append(_predict_baseline(model, dataset, idx))
    return np.mean(preds, axis=0)


def _predict_v0_models(models, dataset: PerturbDataset, idx: np.ndarray, pert_map: dict, ctx_map: dict) -> np.ndarray:
    import torch

    pert_idx_all = np.array([pert_map[p] for p in dataset.obs["pert_id"]], dtype=np.int64)
    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(dataset.X_control[idx], dtype=torch.float32)
            p = torch.as_tensor(pert_idx_all[idx], dtype=torch.long)
            c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long)
            mean, _ = model(x, p, c)
        preds.append(mean.cpu().numpy())
    return np.mean(preds, axis=0)


def _predict_v1_models(models, dataset: PerturbDataset, idx: np.ndarray, pert_gene_masks: dict, ctx_map: dict) -> np.ndarray:
    import torch

    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)
    pert_masks = []
    for pert in dataset.obs["pert_id"]:
        mask = pert_gene_masks.get(pert)
        if mask is None:
            mask = np.zeros(dataset.n_genes, dtype=np.float32)
        pert_masks.append(mask)
    pert_masks = np.stack(pert_masks, axis=0).astype(np.float32)

    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(dataset.X_control[idx], dtype=torch.float32)
            p = torch.as_tensor(pert_masks[idx], dtype=torch.float32)
            c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long)
            mean, _ = model(x, p, c)
        preds.append(mean.cpu().numpy())
    return np.mean(preds, axis=0)


def _predict_v2_models(models, dataset: PerturbDataset, idx: np.ndarray, ctx_map: dict) -> np.ndarray:
    import torch

    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)
    pert_mask = pert_genes_to_mask(dataset.obs.get("pert_genes", [[] for _ in range(dataset.n_obs)]), dataset.var)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            p = torch.as_tensor(pert_mask[idx], dtype=torch.float32)
            c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long)
            mean, _ = model(p, c)
        preds.append(mean.cpu().numpy())
    return np.mean(preds, axis=0)


def _predict_v3_models(models, dataset: PerturbDataset, idx: np.ndarray, ctx_map: dict) -> np.ndarray:
    import torch

    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)
    pert_mask = pert_genes_to_mask(dataset.obs.get("pert_genes", [[] for _ in range(dataset.n_obs)]), dataset.var)
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(dataset.X_control[idx], dtype=torch.float32)
            p = torch.as_tensor(pert_mask[idx], dtype=torch.float32)
            c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long)
            mean, _ = model(x, p, c)
        preds.append(mean.cpu().numpy())
    return np.mean(preds, axis=0)


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
    ensemble_size: int = 1,
    conformal: bool = False,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    def _single():
        model = get_baseline(baseline_name, **kwargs)
        return fit_predict_baseline(model, ds, split)

    if ensemble_size > 1:
        preds = _ensemble_predictions(_single, ensemble_size)
    else:
        preds = _single()

    cfg = {"model": {"name": baseline_name, **kwargs}, "ensemble": ensemble_size, "conformal": conformal}
    cfg_hash = config_hash(cfg)
    run_id = _default_run_id(split_hash, baseline_name, cfg_hash)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)
    if conformal:
        calib_idx, source = _get_calib_idx(split)
        _assert_no_test_leak(calib_idx, split.test_idx)
        if calib_idx.size > 0:
            models = preds.get("models") or [preds.get("model")]
            mean_calib = _predict_baseline_models(models, ds, calib_idx)
            residuals = np.abs(ds.delta[calib_idx] - mean_calib)
            metrics_unc["conformal"] = conformal_intervals(residuals, alphas=[0.5, 0.2, 0.1, 0.05])
            metrics_unc["calib_info"] = {"size": int(calib_idx.size), "source": source}

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "data_hash": _dataset_hash(Path(data_path)),
        "split_hash": split_hash,
        "config_hash": cfg_hash,
        "model": {"name": baseline_name, **kwargs},
        "ensemble": ensemble_size,
        "conformal": conformal,
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
    ensemble_size: int = 1,
    conformal: bool = False,
    batch_size: int | None = None,
    seed: int = 0,
    pretrained_encoder: str | None = None,
    freeze_encoder: bool = False,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    def _single():
        return fit_predict_perturbfm_v0(
            ds,
            split,
            batch_size=batch_size,
            seed=seed,
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

    preds = _ensemble_predictions(_single, ensemble_size) if ensemble_size > 1 else _single()

    cfg = {"model": {"name": "perturbfm_v0", **kwargs}, "ensemble": ensemble_size, "conformal": conformal}
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg["pretrained_encoder"] = pretrained_encoder
    cfg["freeze_encoder"] = freeze_encoder
    cfg_hash = config_hash(cfg)
    run_id = _default_run_id(split_hash, "perturbfm_v0", cfg_hash)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)
    if conformal:
        calib_idx, source = _get_calib_idx(split)
        _assert_no_test_leak(calib_idx, split.test_idx)
        if calib_idx.size > 0:
            models = preds.get("models") or [preds.get("model")]
            mean_calib = _predict_v0_models(models, ds, calib_idx, preds["pert_map"], preds["ctx_map"])
            residuals = np.abs(ds.delta[calib_idx] - mean_calib)
            metrics_unc["conformal"] = conformal_intervals(residuals, alphas=[0.5, 0.2, 0.1, 0.05])
            metrics_unc["calib_info"] = {"size": int(calib_idx.size), "source": source}

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "data_hash": _dataset_hash(Path(data_path)),
        "split_hash": split_hash,
        "config_hash": cfg_hash,
        "model": {"name": "perturbfm_v0", **kwargs},
        "ensemble": ensemble_size,
        "conformal": conformal,
        "batch_size": batch_size,
        "seed": seed,
        "pretrained_encoder": pretrained_encoder,
        "freeze_encoder": freeze_encoder,
    }
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
    ensemble_size: int = 1,
    conformal: bool = False,
    batch_size: int | None = None,
    seed: int = 0,
    pretrained_encoder: str | None = None,
    freeze_encoder: bool = False,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    def _single():
        return fit_predict_perturbfm_v1(
            ds,
            split,
            adjacency=adjacency,
            pert_gene_masks=pert_gene_masks,
            batch_size=batch_size,
            seed=seed,
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

    preds = _ensemble_predictions(_single, ensemble_size) if ensemble_size > 1 else _single()

    cfg = {"model": {"name": "perturbfm_v1", **kwargs}, "ensemble": ensemble_size, "conformal": conformal}
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg["pretrained_encoder"] = pretrained_encoder
    cfg["freeze_encoder"] = freeze_encoder
    cfg_hash = config_hash(cfg)
    run_id = _default_run_id(split_hash, "perturbfm_v1", cfg_hash)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)
    if conformal:
        calib_idx, source = _get_calib_idx(split)
        _assert_no_test_leak(calib_idx, split.test_idx)
        if calib_idx.size > 0:
            models = preds.get("models") or [preds.get("model")]
            mean_calib = _predict_v1_models(models, ds, calib_idx, pert_gene_masks, preds["ctx_map"])
            residuals = np.abs(ds.delta[calib_idx] - mean_calib)
            metrics_unc["conformal"] = conformal_intervals(residuals, alphas=[0.5, 0.2, 0.1, 0.05])
            metrics_unc["calib_info"] = {"size": int(calib_idx.size), "source": source}

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "data_hash": _dataset_hash(Path(data_path)),
        "split_hash": split_hash,
        "config_hash": cfg_hash,
        "model": {"name": "perturbfm_v1", **kwargs},
        "ensemble": ensemble_size,
        "conformal": conformal,
        "batch_size": batch_size,
        "seed": seed,
        "pretrained_encoder": pretrained_encoder,
        "freeze_encoder": freeze_encoder,
    }
    _write_json(run_dir / "config.json", config)
    (run_dir / "split_hash.txt").write_text(split_hash + "\n", encoding="utf-8")

    report_html = render_report(metrics)
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    return run_dir


def run_perturbfm_v2(
    data_path: str,
    split_hash: str,
    adjacency,
    pert_gene_masks=None,
    out_dir: Optional[str] = None,
    ensemble_size: int = 1,
    conformal: bool = False,
    batch_size: int | None = None,
    seed: int = 0,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    def _single():
        return fit_predict_perturbfm_v2(
            ds,
            split,
            adjacencies=adjacency,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )

    preds = _ensemble_predictions(_single, ensemble_size) if ensemble_size > 1 else _single()

    cfg = {"model": {"name": "perturbfm_v2", **kwargs}, "ensemble": ensemble_size, "conformal": conformal}
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg_hash = config_hash(cfg)
    run_id = _default_run_id(split_hash, "perturbfm_v2", cfg_hash)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)
    if conformal:
        calib_idx, source = _get_calib_idx(split)
        _assert_no_test_leak(calib_idx, split.test_idx)
        if calib_idx.size > 0:
            models = preds.get("models") or [preds.get("model")]
            mean_calib = _predict_v2_models(models, ds, calib_idx, preds["ctx_map"])
            residuals = np.abs(ds.delta[calib_idx] - mean_calib)
            metrics_unc["conformal"] = conformal_intervals(residuals, alphas=[0.5, 0.2, 0.1, 0.05])
            metrics_unc["calib_info"] = {"size": int(calib_idx.size), "source": source}

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "data_hash": _dataset_hash(Path(data_path)),
        "split_hash": split_hash,
        "config_hash": cfg_hash,
        "model": {"name": "perturbfm_v2", **kwargs},
        "ensemble": ensemble_size,
        "conformal": conformal,
        "batch_size": batch_size,
        "seed": seed,
    }
    _write_json(run_dir / "config.json", config)
    (run_dir / "split_hash.txt").write_text(split_hash + "\n", encoding="utf-8")

    report_html = render_report(metrics)
    (run_dir / "report.html").write_text(report_html, encoding="utf-8")
    return run_dir


def run_perturbfm_v3(
    data_path: str,
    split_hash: str,
    adjacency,
    out_dir: Optional[str] = None,
    ensemble_size: int = 1,
    conformal: bool = False,
    batch_size: int | None = None,
    seed: int = 0,
    pretrained_encoder: str | None = None,
    freeze_encoder: bool = False,
    **kwargs,
) -> Path:
    ds = PerturbDataset.load_artifact(data_path)
    store = SplitStore.default()
    split = store.load(split_hash)

    def _single():
        return fit_predict_perturbfm_v3(
            ds,
            split,
            adjacencies=adjacency,
            batch_size=batch_size,
            seed=seed,
            pretrained_encoder=pretrained_encoder,
            freeze_encoder=freeze_encoder,
            **kwargs,
        )

    preds = _ensemble_predictions(_single, ensemble_size) if ensemble_size > 1 else _single()

    cfg = {"model": {"name": "perturbfm_v3", **kwargs}, "ensemble": ensemble_size, "conformal": conformal}
    cfg["batch_size"] = batch_size
    cfg["seed"] = seed
    cfg["pretrained_encoder"] = pretrained_encoder
    cfg["freeze_encoder"] = freeze_encoder
    cfg_hash = config_hash(cfg)
    run_id = _default_run_id(split_hash, "perturbfm_v3", cfg_hash)
    run_dir = Path(out_dir) if out_dir else Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(run_dir / "predictions.npz", mean=preds["mean"], var=preds["var"], idx=preds["idx"])

    subset = ds.select(preds["idx"])
    y_true = subset.delta
    metrics_sc = compute_scperturbench_metrics(y_true, preds["mean"], subset.obs)
    metrics_pb = compute_perturbench_metrics(y_true, preds["mean"], subset.obs)
    ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
    metrics_unc = compute_uncertainty_metrics(y_true, preds["mean"], preds["var"], ood_labels=ood_labels)
    if conformal:
        calib_idx, source = _get_calib_idx(split)
        _assert_no_test_leak(calib_idx, split.test_idx)
        if calib_idx.size > 0:
            models = preds.get("models") or [preds.get("model")]
            mean_calib = _predict_v3_models(models, ds, calib_idx, preds["ctx_map"])
            residuals = np.abs(ds.delta[calib_idx] - mean_calib)
            metrics_unc["conformal"] = conformal_intervals(residuals, alphas=[0.5, 0.2, 0.1, 0.05])
            metrics_unc["calib_info"] = {"size": int(calib_idx.size), "source": source}

    metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
    _require_metrics_complete(metrics)
    _write_json(run_dir / "metrics.json", metrics)
    _write_json(run_dir / "calibration.json", metrics_unc)

    config = {
        "data_path": data_path,
        "data_hash": _dataset_hash(Path(data_path)),
        "split_hash": split_hash,
        "config_hash": cfg_hash,
        "model": {"name": "perturbfm_v3", **kwargs},
        "ensemble": ensemble_size,
        "conformal": conformal,
        "batch_size": batch_size,
        "seed": seed,
        "pretrained_encoder": pretrained_encoder,
        "freeze_encoder": freeze_encoder,
    }
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
