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
import sys
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


def _external_missing(name: str, path: Path) -> None:
    print(f"[info] {name} reference repo not found at {path}.")
    print(f"[info] Clone it into {path} (gitignored) to enable parity checks.")


def _extract_panel(ref: dict, panel: str) -> dict | None:
    if panel in ref:
        return ref.get(panel)
    if "global" in ref:
        return ref
    return None


def compare_globals(ours: dict, ref: dict, panel: str, tol: float) -> bool:
    ours_panel = _extract_panel(ours, panel)
    ref_panel = _extract_panel(ref, panel)
    if ours_panel is None or ref_panel is None:
        print(f"[warn] missing panel {panel} for comparison")
        return False
    ours_global = ours_panel.get("global", ours_panel)
    ref_global = ref_panel.get("global", ref_panel)
    ok = True
    for key, val in ours_global.items():
        ref_val = ref_global.get(key)
        if ref_val is None:
            print(f"[warn] reference missing {panel}.global.{key}")
            ok = False
            continue
        if np.isnan(val) and np.isnan(ref_val):
            continue
        if abs(ref_val - val) > tol:
            print(f"[fail] {panel}.global.{key}: ours={val:.6f} ref={ref_val:.6f}")
            ok = False
    if ok:
        print(f"[ok] {panel} comparison within tolerance")
    return ok


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
    exit_code = 0

    # Optional reference comparison
    if args.reference_json:
        ref = json.loads(Path(args.reference_json).read_text())
        ok_sc = compare_globals(ours, ref, "scperturbench", args.tol)
        ok_pb = compare_globals(ours, ref, "perturbench", args.tol)
        if not (ok_sc and ok_pb):
            exit_code = 1

    # Optional external comparisons (expected to be implemented by user locally)
    scpb = maybe_run_external(
        Path("third_party/scPerturBench"),
        ["python", "reference_metrics.py", str(data_path), str(preds_path)],
    )
    if scpb:
        print("=== scPerturBench reference metrics ===")
        print(json.dumps(scpb, indent=2))
        if not compare_globals(ours, scpb, "scperturbench", args.tol):
            exit_code = 1
    else:
        _external_missing("scPerturBench", Path("third_party/scPerturBench"))

    pb = maybe_run_external(
        Path("third_party/PerturBench"),
        ["python", "reference_metrics.py", str(data_path), str(preds_path)],
    )
    if pb:
        print("=== PerturBench reference metrics ===")
        print(json.dumps(pb, indent=2))
        if not compare_globals(ours, pb, "perturbench", args.tol):
            exit_code = 1
    else:
        _external_missing("PerturBench", Path("third_party/PerturBench"))

    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
