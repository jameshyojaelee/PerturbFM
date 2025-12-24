#!/usr/bin/env python3
"""
Train GEARS on a dataset and export predictions in PerturbFM format.

This uses a perturbation-level custom split derived from a SplitStore hash.
Intended for perturbation-OOD splits (e.g., Norman19).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_store import SplitStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adata", required=True, help="Input AnnData (.h5ad) for GEARS.")
    ap.add_argument("--artifact", required=True, help="PerturbDataset artifact dir.")
    ap.add_argument("--split", required=True, help="Split hash (test perts derived).")
    ap.add_argument("--workdir", required=True, help="Working directory for GEARS processing.")
    ap.add_argument("--out-preds", required=True, help="Output predictions.npz path.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--hidden-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    try:
        import scanpy as sc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scanpy required for GEARS") from exc
    from gears import PertData, GEARS
    import gears.pertdata as gears_pertdata
    import gears.utils as gears_utils
    import anndata as ad

    ds = PerturbDataset.load_artifact(args.artifact)
    store = SplitStore.default()
    split = store.load(args.split)

    train_perts = sorted(set(ds.obs["pert_id"][i] for i in split.train_idx) - {"control"})
    val_perts = sorted(set(ds.obs["pert_id"][i] for i in split.val_idx) - {"control"})
    test_perts = sorted(set(ds.obs["pert_id"][i] for i in split.test_idx) - {"control"})
    if "ctrl" not in train_perts:
        train_perts.append("ctrl")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    split_path = workdir / "gears_split.pkl"
    with split_path.open("wb") as f:
        pickle.dump({"train": train_perts, "val": val_perts, "test": test_perts}, f)

    adata = sc.read_h5ad(args.adata)
    try:
        import scipy.sparse as sp
        if not sp.issparse(adata.X):
            adata.X = sp.csr_matrix(adata.X)
    except Exception:
        pass

    # Patch GEARS GO-graph filter to handle single-gene perturbations (no '+')
    def _filter_pert_in_go_safe(condition, pert_names):
        if condition == "ctrl":
            return True
        parts = str(condition).split("+")
        if len(parts) == 1:
            cond1, cond2 = parts[0], "ctrl"
        else:
            cond1, cond2 = parts[0], parts[1]
        num_ctrl = (cond1 == "ctrl") + (cond2 == "ctrl")
        num_in_perts = (cond1 in pert_names) + (cond2 in pert_names)
        return (num_ctrl + num_in_perts) == 2

    gears_utils.filter_pert_in_go = _filter_pert_in_go_safe
    gears_pertdata.filter_pert_in_go = _filter_pert_in_go_safe

    pert_data = PertData(str(workdir))
    pert_data.new_data_process(dataset_name="pfm_gears", adata=adata)
    pert_data.load(data_path=str(workdir / "pfm_gears"))
    pert_data.prepare_split(split="custom", split_dict_path=str(split_path))
    pert_data.get_dataloader(batch_size=args.batch_size, test_batch_size=128)

    gears_model = GEARS(pert_data, device=args.device)
    gears_model.model_initialize(hidden_size=args.hidden_size)
    gears_model.train(epochs=args.epochs)

    pert_list = [p.split("+") for p in test_perts]
    preds = gears_model.predict(pert_list)

    # Build adata_pred with one row per perturbation (mean prediction)
    ctx = str(ds.obs["context_id"][0])
    batch = str(ds.obs["batch_id"][0])
    rows = []
    obs = {"condition": [], "cell_type": [], "batch_id": []}
    for pert in test_perts:
        key = pert.replace("+", "_")
        vec = preds[key]
        rows.append(vec)
        obs["condition"].append(pert)
        obs["cell_type"].append(ctx)
        obs["batch_id"].append(batch)
    X = np.vstack(rows).astype(np.float32)
    var = ad.AnnData(X=ds.X_pert[:1]).var.copy()
    var["gene_name"] = list(ds.var)
    adata_pred = ad.AnnData(X=X, obs=obs, var=var)

    pred_path = workdir / "gears_adata_pred.h5ad"
    adata_pred.write_h5ad(pred_path)

    import runpy
    import sys

    # Call converter via CLI-like invocation
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
