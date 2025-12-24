#!/usr/bin/env python3
"""
Export a PerturbDataset artifact to an AnnData .h5ad for external models.

This keeps row order identical to the artifact, so SplitStore indices (idx)
remain valid for training/validation/test selection and for exporting predictions.

Usage:
  PYTHONPATH=src python3 scripts/external/export_artifact_to_h5ad.py \
    --data data/artifacts/perturbench/norman19 \
    --out data/external/norman19.h5ad
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset


def _ensure_str_list(values):
    return [str(v) for v in values]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PerturbDataset artifact directory.")
    ap.add_argument("--out", required=True, help="Output .h5ad path.")
    ap.add_argument("--use-x", choices=["X_pert", "X_control", "delta"], default="X_pert", help="What to store as adata.X.")
    args = ap.parse_args()

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anndata is required; install extras with: pip install -e '.[bench]'") from exc

    ds = PerturbDataset.load_artifact(args.data)

    X_map = {"X_pert": ds.X_pert, "X_control": ds.X_control, "delta": ds.delta}
    X = X_map.get(args.use_x)
    if X is None:
        raise ValueError(f"Requested use-x={args.use_x} but that array is missing in the artifact.")

    obs = {
        "pert_id": _ensure_str_list(ds.obs.get("pert_id", [""] * ds.n_obs)),
        "context_id": _ensure_str_list(ds.obs.get("context_id", [""] * ds.n_obs)),
        "batch_id": _ensure_str_list(ds.obs.get("batch_id", [""] * ds.n_obs)),
        "is_control": np.asarray(ds.obs.get("is_control", [False] * ds.n_obs), dtype=bool),
    }
    # External repos often assume these conventional keys:
    # - GEARS: obs['condition'] with 'ctrl' as control label; obs['cell_type']
    pert = np.array(obs["pert_id"], dtype=object)
    obs["condition"] = ["ctrl" if p == "control" else str(p) for p in pert]
    obs["cell_type"] = obs["context_id"]
    # CPA-style keys
    obs["perturbation"] = obs["condition"]
    # Dummy dosage for gene/drug perturbations: one 1.0 per component
    dose = []
    for p in obs["condition"]:
        if p == "ctrl":
            dose.append("0.0")
        else:
            parts = str(p).split("+")
            dose.append("+".join(["1.0"] * len(parts)))
    obs["dosage"] = dose

    var = {"gene_name": _ensure_str_list(ds.var)}
    adata = ad.AnnData(X=X, obs=obs, var=var)
    # Preserve additional arrays as layers for models that want them
    if ds.X_control is not None:
        adata.layers["X_control"] = ds.X_control.astype(np.float32, copy=False)
    if ds.X_pert is not None:
        adata.layers["X_pert"] = ds.X_pert.astype(np.float32, copy=False)
    if ds.delta is not None:
        adata.layers["delta"] = ds.delta.astype(np.float32, copy=False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Wrote {out_path} (n_obs={adata.n_obs}, n_vars={adata.n_vars})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

