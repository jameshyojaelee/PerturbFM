#!/usr/bin/env python3
"""
Convert a dataset artifact (or .h5ad) into a memmap-backed artifact.

Usage:
  python scripts/convert_to_memmap.py --data /path/to/artifact --out /path/to/memmap_artifact
"""

from __future__ import annotations

import argparse
from pathlib import Path

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.adapters.perturbench import PerturBenchAdapter


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Artifact directory or .h5ad path.")
    ap.add_argument("--out", required=True, help="Output directory for memmap artifact.")
    ap.add_argument("--backed", action="store_true", help="Use backed mode when reading .h5ad.")
    args = ap.parse_args()

    data_path = Path(args.data)
    if data_path.suffix == ".h5ad":
        adapter = PerturBenchAdapter(data_path)
        ds = adapter.load(backed=args.backed)
    else:
        ds = PerturbDataset.load_artifact(data_path)

    ds.save_memmap_artifact(args.out)
    print(f"Wrote memmap artifact to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
