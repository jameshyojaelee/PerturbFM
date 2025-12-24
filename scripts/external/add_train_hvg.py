#!/usr/bin/env python3
"""
Add a train-only HVG embedding matrix to an AnnData file.

This is intended for external models (e.g., STATE) that expect an embedding key
like `.obsm['X_hvg']`.

We compute HVGs using *train_idx only* from a frozen split hash to avoid leakage.

Usage:
  PYTHONNOUSERSITE=1 PYTHONPATH=src python3 scripts/external/add_train_hvg.py \
    --adata data/external/norman19.h5ad \
    --split da58b88d99c4133b173cf75d4998649a36974d7ec4ebd831930a10328cfe0b1b \
    --out data/external/norman19_hvg2000.h5ad \
    --n-hvgs 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.splits.split_store import SplitStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="Input .h5ad path.")
    ap.add_argument("--split", required=True, help="Split hash (train_idx used).")
    ap.add_argument("--out", required=True, help="Output .h5ad path.")
    ap.add_argument("--n-hvgs", type=int, default=2000)
    ap.add_argument("--embed-key", default="X_hvg")
    args = ap.parse_args()

    try:
        import anndata as ad
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("anndata is required") from exc

    store = SplitStore.default()
    split = store.load(args.split)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)

    adata = ad.read_h5ad(args.adata)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    X_tr = X[train_idx]
    # simple HVG proxy: variance across train only
    var = X_tr.var(axis=0)
    n = min(int(args.n_hvgs), var.shape[0])
    hvg_idx = np.argpartition(var, -n)[-n:]
    hvg_idx = np.sort(hvg_idx)

    adata.var["highly_variable"] = False
    adata.var.loc[adata.var.index[hvg_idx], "highly_variable"] = True
    adata.obsm[args.embed_key] = X[:, hvg_idx].astype(np.float32, copy=False)
    adata.uns.setdefault("hvg_info", {})[args.embed_key] = {
        "n_hvgs": int(n),
        "split_hash": args.split,
        "method": "train_variance_topk",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Wrote {out_path} with obsm['{args.embed_key}'] shape={adata.obsm[args.embed_key].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

