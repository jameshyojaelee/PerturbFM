#!/usr/bin/env python3
"""
Export predictions for external benchmark harnesses.

Usage:
  python scripts/export_predictions.py --preds runs/<run_id>/predictions.npz --out /tmp/preds --data /path/to/artifact

Outputs:
  - predictions.npz (mean/var/idx)
  - predictions.csv (optional) with header = gene names if data artifact provided
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Path to predictions.npz")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--data", help="Optional dataset artifact dir (for gene names)")
    ap.add_argument("--format", choices=["npz", "csv", "both"], default="both")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.preds)
    mean = npz["mean"]
    var = npz["var"]
    idx = npz["idx"]

    if args.format in ("npz", "both"):
        np.savez_compressed(out_dir / "predictions.npz", mean=mean, var=var, idx=idx)

    if args.format in ("csv", "both"):
        header = None
        if args.data:
            ds = PerturbDataset.load_artifact(args.data)
            header = ["idx"] + list(ds.var)
        rows = np.concatenate([idx[:, None], mean], axis=1)
        fmt = ["%d"] + ["%.6g"] * mean.shape[1]
        np.savetxt(out_dir / "predictions.csv", rows, delimiter=",", header=",".join(header) if header else "", comments="", fmt=fmt)

    print(f"Wrote exports to {out_dir}")


if __name__ == "__main__":
    main()
