#!/usr/bin/env python3
"""
Import scPerturBench data into PerturbDataset artifact format.

Usage:
  python scripts/import_scperturbench.py --dataset <name_or_path> --out /path/to/artifact
"""

from __future__ import annotations

import argparse

from perturbfm.data.adapters.scperturbench import ScPerturBenchAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="scPerturBench dataset name or path.")
    ap.add_argument("--out", required=True, help="Output artifact directory.")
    ap.add_argument("--backed", action="store_true", help="Use backed mode when reading .h5ad.")
    args = ap.parse_args()

    adapter = ScPerturBenchAdapter(args.dataset)
    ds = adapter.load(backed=args.backed)
    ds.save_artifact(args.out)
    print(f"Wrote scPerturBench artifact to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
