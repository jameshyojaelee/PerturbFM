#!/usr/bin/env python3
"""
Build a co-expression graph prior from training data only.

Usage:
  python scripts/build_coexpr_graph.py \
    --data data/artifacts/perturbench/norman19 \
    --split <SPLIT_HASH> \
    --out graphs/norman19_coexpr_train_top20.npz \
    --max-cells 5000 \
    --topk 20 \
    --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return mat / row_sum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="PerturbDataset artifact path.")
    ap.add_argument("--split", required=True, help="Split hash (train indices only).")
    ap.add_argument("--out", required=True, help="Output .npz path with key 'adjacency'.")
    ap.add_argument("--max-cells", type=int, default=5000, help="Max train cells to use.")
    ap.add_argument("--topk", type=int, default=20, help="Top-k neighbors per gene by correlation.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use-control", action="store_true", help="Use X_control if available (default).")
    ap.add_argument("--allow-negative", action="store_true", help="Keep negative correlations (default: drop).")
    args = ap.parse_args()

    ds = PerturbDataset.load_artifact(args.data)
    store = SplitStore.default()
    split = store.load(args.split)

    X = ds.X_control if (args.use_control or ds.X_pert is None) else ds.X_pert
    if X is None:
        raise ValueError("Dataset must include X_control or X_pert.")

    rng = np.random.default_rng(args.seed)
    train_idx = np.asarray(split.train_idx, dtype=np.int64)
    if args.max_cells > 0 and train_idx.size > args.max_cells:
        train_idx = rng.choice(train_idx, size=args.max_cells, replace=False)

    X = X[train_idx].astype(np.float32, copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = X / std

    corr = (X.T @ X) / max(1, X.shape[0] - 1)
    corr = corr.astype(np.float32, copy=False)
    np.fill_diagonal(corr, 0.0)
    if not args.allow_negative:
        corr[corr < 0] = 0.0

    g = corr.shape[0]
    topk = min(args.topk, g - 1) if g > 1 else 0
    adjacency = np.zeros_like(corr, dtype=np.float32)
    if topk > 0:
        for i in range(g):
            row = corr[i]
            idx = np.argpartition(row, -topk)[-topk:]
            weights = row[idx]
            adjacency[i, idx] = weights

    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 1.0)
    adjacency = _row_normalize(adjacency)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, adjacency=adjacency, genes=np.array(ds.var, dtype=object))

    notes = out_path.with_suffix(".notes.txt")
    density = float((adjacency > 0).sum()) / float(g * g)
    notes.write_text(
        "\n".join(
            [
                f"data={args.data}",
                f"split={args.split}",
                f"max_cells={args.max_cells}",
                f"topk={args.topk}",
                f"seed={args.seed}",
                f"allow_negative={args.allow_negative}",
                f"n_genes={g}",
                f"density={density:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote graph to {out_path}")
    print(f"Wrote notes to {notes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
