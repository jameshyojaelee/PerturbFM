#!/usr/bin/env python3
"""
Convert an external model's predicted-expression AnnData into PerturbFM predictions.npz.

This expects an AnnData where:
- adata_pred.X is predicted expression in the same gene space as the artifact var.
- adata_pred.obs contains keys for perturbation/context/batch (strings).

We assign predictions to test_idx rows by matching (pert_id, context_id, batch_id) and
using the *group mean* predicted expression for that triple. This avoids requiring
row-wise identity / barcode alignment across codebases.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore


def _to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PerturbDataset artifact dir.")
    ap.add_argument("--split", required=True, help="Split hash (test_idx used).")
    ap.add_argument("--adata-pred", required=True, help="Path to predicted-expression .h5ad.")
    ap.add_argument("--pert-col", default="condition", help="Column in adata_pred.obs containing perturbation label.")
    ap.add_argument("--celltype-col", default="cell_type", help="Column in adata_pred.obs containing context/cell type.")
    ap.add_argument("--batch-col", default="batch_id", help="Column in adata_pred.obs containing batch.")
    ap.add_argument("--out", required=True, help="Output predictions.npz path.")
    ap.add_argument("--var-eps", type=float, default=1e-6, help="Fallback variance if external var is unavailable.")
    args = ap.parse_args()

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anndata is required") from exc

    ds = PerturbDataset.load_artifact(args.data)
    store = SplitStore.default()
    split = store.load(args.split)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)

    adata_pred = ad.read_h5ad(args.adata_pred)
    X_pred = _to_dense(adata_pred.X).astype(np.float32, copy=False)

    # Align gene order if possible
    if adata_pred.var is not None:
        if "gene_names" in adata_pred.var.columns:
            pred_genes = list(adata_pred.var["gene_names"].astype(str).values)
        elif "gene_name" in adata_pred.var.columns:
            pred_genes = list(adata_pred.var["gene_name"].astype(str).values)
        else:
            pred_genes = list(map(str, getattr(adata_pred.var, "index", [])))
        if pred_genes and len(pred_genes) == X_pred.shape[1]:
            gene_to_idx = {g: i for i, g in enumerate(pred_genes)}
            cols = [gene_to_idx.get(g) for g in ds.var]
            if all(c is not None for c in cols):
                X_pred = X_pred[:, np.array(cols, dtype=np.int64)]

    obs = adata_pred.obs
    keys = (obs[args.pert_col].astype(str), obs[args.celltype_col].astype(str), obs[args.batch_col].astype(str))

    group_sum: Dict[Tuple[str, str, str], np.ndarray] = {}
    group_n: Dict[Tuple[str, str, str], int] = {}
    for i in range(adata_pred.n_obs):
        k = (keys[0].iat[i], keys[1].iat[i], keys[2].iat[i])
        if k not in group_sum:
            group_sum[k] = X_pred[i].copy()
            group_n[k] = 1
        else:
            group_sum[k] += X_pred[i]
            group_n[k] += 1

    group_mean: Dict[Tuple[str, str, str], np.ndarray] = {k: v / float(group_n[k]) for k, v in group_sum.items()}

    # Build predictions for test rows
    mean = np.zeros((test_idx.size, ds.n_genes), dtype=np.float32)
    for out_i, idx in enumerate(test_idx):
        pert = str(ds.obs["pert_id"][idx])
        ctx = str(ds.obs["context_id"][idx])
        batch = str(ds.obs["batch_id"][idx])
        cond = "ctrl" if pert == "control" else pert
        k = (cond, ctx, batch)
        pred_expr = group_mean.get(k)
        if pred_expr is None:
            # fallback: predict no change
            pred_expr = ds.X_control[idx]
        mean[out_i] = pred_expr - ds.X_control[idx]

    var = np.full_like(mean, float(args.var_eps), dtype=np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mean=mean, var=var, idx=test_idx)
    print(f"Wrote {out_path} with mean shape={mean.shape} idx shape={test_idx.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

