#!/usr/bin/env python3
"""
Train scGen on a dataset and export predictions in PerturbFM format.

Intended for context-OOD splits where perturbations are seen in training contexts.
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
    ap.add_argument("--holdout-context", required=True, help="Context ID held out for test.")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=256)
    ap.add_argument("--n-latent", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    args = ap.parse_args()

    import scanpy as sc
    import scgen
    import anndata as ad

    ds = PerturbDataset.load_artifact(args.artifact)
    store = SplitStore.default()
    split = store.load(args.split)
    test_idx = np.asarray(split.test_idx, dtype=np.int64)
    test_perts = sorted(set(ds.obs["pert_id"][i] for i in test_idx) - {"control"})

    adata = sc.read_h5ad(args.adata)
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    # Train on contexts excluding holdout
    train_mask = adata.obs["cell_type"] != str(args.holdout_context)
    adata_train = adata[train_mask].copy()

    scgen.SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="cell_type")
    model = scgen.SCGEN(adata_train, n_hidden=args.n_hidden, n_latent=args.n_latent, n_layers=args.n_layers)
    model.train(max_epochs=args.max_epochs, batch_size=args.batch_size)

    # Control cells from holdout context for prediction
    ctrl_target = adata[(adata.obs["cell_type"] == str(args.holdout_context)) & (adata.obs["condition"] == "ctrl")].copy()
    if ctrl_target.n_obs == 0:
        raise ValueError("No control cells found in holdout context.")
    ctrl_manager = model.adata_manager.transfer_fields(ctrl_target, extend_categories=True)
    model._register_manager_for_instance(ctrl_manager)

    # Predict per perturbation, use mean expression per pert
    rows = []
    obs = {"condition": [], "cell_type": [], "batch_id": []}
    batch_id = str(ds.obs["batch_id"][0])
    for pert in test_perts:
        pred_adata, _ = model.predict(ctrl_key="ctrl", stim_key=pert, adata_to_predict=ctrl_target)
        rows.append(np.asarray(pred_adata.X).mean(axis=0))
        obs["condition"].append(pert)
        obs["cell_type"].append(str(args.holdout_context))
        obs["batch_id"].append(batch_id)

    X = np.vstack(rows).astype(np.float32)
    var = ad.AnnData(X=ds.X_pert[:1]).var.copy()
    var["gene_name"] = list(ds.var)
    adata_pred = ad.AnnData(X=X, obs=obs, var=var)
    pred_path = Path(args.out_preds).with_suffix(".adata_pred.h5ad")
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    adata_pred.write_h5ad(pred_path)

    import runpy
    import sys

    sys.argv = [
        "convert",
        "--data",
        args.artifact,
        "--split",
        args.split,
        "--adata-pred",
        str(pred_path),
        "--pert-col",
        "condition",
        "--celltype-col",
        "cell_type",
        "--batch-col",
        "batch_id",
        "--out",
        args.out_preds,
    ]
    convert_path = Path(__file__).resolve().parent / "convert_adata_pred_to_predictions.py"
    runpy.run_path(str(convert_path), run_name="__main__")

    print(f"Wrote predictions to {args.out_preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
