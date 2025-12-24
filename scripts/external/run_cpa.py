#!/usr/bin/env python3
"""
Train CPA on a dataset and export predictions in PerturbFM format.

Intended for context-OOD where perturbations are seen in training contexts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="Input AnnData (.h5ad).")
    ap.add_argument("--artifact", required=True, help="PerturbDataset artifact dir.")
    ap.add_argument("--split", required=True, help="Split hash.")
    ap.add_argument("--out-preds", required=True, help="Output predictions.npz path.")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    import scanpy as sc
    import cpa

    ds = PerturbDataset.load_artifact(args.artifact)
    store = SplitStore.default()
    split = store.load(args.split)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)

    adata = sc.read_h5ad(args.adata)
    # Build split labels in obs
    adata.obs["split"] = "train"
    adata.obs.iloc[split.val_idx, adata.obs.columns.get_loc("split")] = "val"
    adata.obs.iloc[split.test_idx, adata.obs.columns.get_loc("split")] = "test"

    # CPA setup
    cpa.CPA.setup_anndata(
        adata,
        perturbation_key="condition",
        control_group="ctrl",
        dosage_key="dosage",
        batch_key="batch_id",
        categorical_covariate_keys=["cell_type"],
        is_count_data=False,
    )

    model = cpa.CPA(adata, split_key="split", train_split="train", valid_split="val", test_split="test")
    model.train(
        max_epochs=args.max_epochs,
        use_gpu=True,
        batch_size=args.batch_size,
        early_stopping_patience=5,
        check_val_every_n_epoch=5,
    )

    # Counterfactual prediction: set X to control baseline
    if "X_control" in adata.layers:
        adata.X = adata.layers["X_control"].copy()
    model.predict(adata=adata, indices=test_idx, batch_size=args.batch_size, n_samples=1)
    pred = adata.obsm["CPA_pred"][test_idx]

    mean = pred - ds.X_control[test_idx]
    var = np.full_like(mean, 1e-6, dtype=np.float32)

    out_path = Path(args.out_preds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, mean=mean, var=var, idx=test_idx)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

