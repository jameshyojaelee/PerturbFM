#!/usr/bin/env python3
"""Materialize a Tahoe-100M subset into a PerturbDataset artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.tahoe import TahoeConfig, iter_tahoe_batches, load_metadata_table
from perturbfm.utils.hashing import stable_json_dumps, sha256_json


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Tahoe-100M root dir.")
    ap.add_argument("--out", required=True, help="Output artifact directory.")
    ap.add_argument("--max-cells", type=int, default=50000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle-files", action="store_true")
    ap.add_argument("--manifest", required=False, help="Optional shard manifest JSON for provenance.")
    args = ap.parse_args()

    cfg = TahoeConfig(root=Path(args.root), batch_size=args.batch_size, seed=args.seed, shuffle_files=args.shuffle_files)

    X = []
    obs = {k: [] for k in ("pert_id", "context_id", "batch_id", "is_control")}
    var = None
    n = 0

    for batch in iter_tahoe_batches(cfg):
        expr = batch["expressions"]
        genes = batch["genes"]
        if var is None:
            first_genes = genes[0] if isinstance(genes, list) else list(genes)[0]
            var = [str(g) for g in first_genes]
        # Tahoe data is not explicitly split into control vs pert in raw form; use drug as pert_id.
        drugs = batch.get("drug") or ["unknown"] * len(expr)
        cell_lines = batch.get("cell_line_id") or ["unknown"] * len(expr)
        for i in range(len(expr)):
            X.append(expr[i])
            obs["pert_id"].append(str(drugs[i]))
            obs["context_id"].append(str(cell_lines[i]))
            obs["batch_id"].append("tahoe")
            obs["is_control"].append(False)
            n += 1
            if n >= args.max_cells:
                break
        if n >= args.max_cells:
            break

    X = np.asarray(X, dtype=np.float32)
    if var is None:
        raise ValueError("No data loaded from Tahoe shards.")

    metadata = {
        "source": "tahoe-100m",
        "root": str(Path(args.root)),
        "max_cells": args.max_cells,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    if args.manifest:
        manifest_text = Path(args.manifest).read_text()
        metadata["manifest_hash"] = sha256_json(manifest_text)
        metadata["manifest_path"] = args.manifest

    # Save as X_pert only; delta is not defined for raw expression.
    ds = PerturbDataset(X_control=None, X_pert=X, delta=None, obs=obs, var=var, metadata=metadata)
    ds.validate()
    out_dir = Path(args.out)
    ds.save_artifact(out_dir)
    (out_dir / "tahoe_info.json").write_text(stable_json_dumps(metadata), encoding="utf-8")
    print(f"Wrote Tahoe subset artifact to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
