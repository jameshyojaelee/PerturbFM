#!/usr/bin/env python3
"""
Evaluate external model predictions against PerturbFM metrics.

Manifest format (JSON):
[
  {
    "name": "STATE",
    "data": "data/artifacts/perturbench/norman19",
    "split": "<SPLIT_HASH>",
    "preds": "/path/to/external/predictions.npz"
  }
]

Each predictions.npz must contain:
- mean: [N, G]
- idx: [N] indices into the dataset
- var: [N, G] (optional; if missing, a tiny constant variance is used and noted)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics
from perturbfm.eval.uncertainty_metrics import compute_uncertainty_metrics
from perturbfm.eval.report import render_report
from perturbfm.eval.evaluator import _require_metrics_complete
from perturbfm.utils.hashing import stable_json_dumps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to external predictions manifest JSON.")
    ap.add_argument("--out-root", default="runs/external_eval", help="Root directory for evaluation outputs.")
    ap.add_argument("--skip-missing", action="store_true", help="Skip entries with missing predictions.")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for entry in manifest:
        name = entry["name"]
        data_path = entry["data"]
        split_hash = entry["split"]
        preds_path = Path(entry["preds"])
        if not preds_path.exists():
            msg = f"[warn] missing predictions for {name}: {preds_path}"
            print(msg)
            if args.skip_missing:
                continue
            raise FileNotFoundError(msg)

        ds = PerturbDataset.load_artifact(data_path)
        store = SplitStore.default()
        _ = store.load(split_hash)

        npz = np.load(preds_path)
        if "mean" not in npz or "idx" not in npz:
            raise ValueError(f"{preds_path} must include mean and idx arrays.")
        mean = npz["mean"]
        idx = npz["idx"]
        if "var" in npz:
            var = npz["var"]
            var_note = None
        else:
            var = np.full_like(mean, 1e-6, dtype=np.float32)
            var_note = "var missing; filled with constant 1e-6 for uncertainty metrics."

        subset = ds.select(idx)
        y_true = subset.delta

        metrics_sc = compute_scperturbench_metrics(y_true, mean, subset.obs)
        metrics_pb = compute_perturbench_metrics(y_true, mean, subset.obs)
        ood_labels = subset.obs.get("is_ood") if isinstance(subset.obs, dict) else None
        metrics_unc = compute_uncertainty_metrics(y_true, mean, var, ood_labels=ood_labels)
        if var_note:
            metrics_unc.setdefault("notes", {})["var_fill"] = var_note

        metrics = {"scperturbench": metrics_sc, "perturbench": metrics_pb, "uncertainty": metrics_unc}
        _require_metrics_complete(metrics)

        ds_name = Path(data_path).name
        out_dir = out_root / name / ds_name / split_hash
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(stable_json_dumps(metrics), encoding="utf-8")
        (out_dir / "calibration.json").write_text(stable_json_dumps(metrics_unc), encoding="utf-8")
        (out_dir / "report.html").write_text(render_report(metrics), encoding="utf-8")
        meta = {"name": name, "data": data_path, "split": split_hash, "preds": str(preds_path)}
        if var_note:
            meta["notes"] = var_note
        (out_dir / "meta.json").write_text(stable_json_dumps(meta), encoding="utf-8")

        results.append({"name": name, "data": data_path, "split": split_hash, "out_dir": str(out_dir)})
        print(f"[ok] wrote {out_dir}")

    if results:
        (out_root / "manifest_results.json").write_text(stable_json_dumps(results), encoding="utf-8")
        print(f"Wrote manifest results to {out_root / 'manifest_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
