#!/usr/bin/env python3
"""
Numerical parity harness for metrics.

Usage:
  python scripts/validate_metrics.py --data <artifact_dir> --preds <predictions.npz>
Optionally, if third_party/scPerturBench or third_party/PerturBench is present,
this script will attempt to call their reference metric scripts for comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.eval.metrics_scperturbench import compute_scperturbench_metrics
from perturbfm.eval.metrics_perturbench import compute_perturbench_metrics
from perturbfm.eval.uncertainty_metrics import compute_uncertainty_metrics
from perturbfm.eval.evaluator import _require_metrics_complete
from perturbfm.utils.hashing import stable_json_dumps


def run_internal(data_path: Path, preds_path: Path):
    ds = PerturbDataset.load_artifact(data_path)
    npz = np.load(preds_path)
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
    return metrics


def maybe_run_external(tool_dir: Path, cmd: list[str]) -> dict | None:
    if not tool_dir.exists():
        return None
    try:
        out = subprocess.check_output(cmd, cwd=tool_dir, text=True)
        return json.loads(out)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to dataset artifact directory")
    ap.add_argument("--preds", required=True, help="Path to predictions.npz")
    ap.add_argument("--reference-json", help="Optional reference metrics JSON to compare against.")
    ap.add_argument("--tol", type=float, default=1e-3, help="Tolerance for numeric comparisons.")
    args = ap.parse_args()

    data_path = Path(args.data)
    preds_path = Path(args.preds)

    ours = run_internal(data_path, preds_path)
    print("=== PerturbFM metrics ===")
    print(stable_json_dumps(ours))

    # Optional reference comparison
    if args.reference_json:
        ref = json.loads(Path(args.reference_json).read_text())
        ok = True
        for panel in ("scperturbench", "perturbench"):
            for key, val in ours[panel]["global"].items():
                ref_val = ref.get(panel, {}).get("global", {}).get(key)
                if ref_val is None:
                    print(f"[warn] reference missing {panel}.global.{key}")
                    ok = False
                    continue
                if abs(ref_val - val) > args.tol:
                    print(f"[fail] {panel}.global.{key}: ours={val:.6f} ref={ref_val:.6f}")
                    ok = False
        if ok:
            print("[ok] reference comparison within tolerance")

    # Optional external comparisons (expected to be implemented by user locally)
    scpb = maybe_run_external(
        Path("third_party/scPerturBench"),
        ["python", "reference_metrics.py", str(data_path), str(preds_path)],
    )
    if scpb:
        print("=== scPerturBench reference metrics ===")
        print(json.dumps(scpb, indent=2))

    pb = maybe_run_external(
        Path("third_party/PerturBench"),
        ["python", "reference_metrics.py", str(data_path), str(preds_path)],
    )
    if pb:
        print("=== PerturBench reference metrics ===")
        print(json.dumps(pb, indent=2))


if __name__ == "__main__":
    main()
