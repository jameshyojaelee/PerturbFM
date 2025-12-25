"""Command-line interface for PerturbFM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_doctor(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("doctor", help="Basic environment checks.")
    parser.set_defaults(func=_cmd_doctor)


def _add_config(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("config", help="Config utilities (stub).")
    parser.set_defaults(func=_cmd_config)


def _add_data(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("data", help="Dataset utilities.")
    data_sub = parser.add_subparsers(dest="data_command", required=True)

    make_synth = data_sub.add_parser("make-synth", help="Write a synthetic dataset artifact.")
    make_synth.add_argument("--out", required=True, help="Output directory for the dataset artifact.")
    make_synth.add_argument("--n-obs", type=int, default=200)
    make_synth.add_argument("--n-genes", type=int, default=50)
    make_synth.add_argument("--n-contexts", type=int, default=3)
    make_synth.add_argument("--n-perts", type=int, default=5)
    make_synth.add_argument("--seed", type=int, default=0)
    make_synth.set_defaults(func=_cmd_data_make_synth)

    info = data_sub.add_parser("info", help="Inspect a dataset artifact.")
    info.add_argument("path", help="Path to dataset artifact directory.")
    info.set_defaults(func=_cmd_data_info)

    import_pb = data_sub.add_parser("import-perturbench", help="Import a PerturBench dataset into PerturbDataset artifact format.")
    import_pb.add_argument("--dataset", required=True, help="PerturBench dataset name or path.")
    import_pb.add_argument("--out", required=True, help="Output directory for the dataset artifact.")
    import_pb.add_argument("--backed", action="store_true", help="Use backed mode when reading .h5ad (still materializes arrays).")
    import_pb.set_defaults(func=_cmd_data_import_perturbench)


def _add_splits(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("splits", help="Split utilities.")
    split_sub = parser.add_subparsers(dest="split_command", required=True)

    create = split_sub.add_parser("create", help="Create and store a split.")
    create.add_argument("--data", required=True, help="Path to dataset artifact directory.")
    create.add_argument(
        "--spec",
        required=True,
        choices=["context_ood", "perturbation_ood", "combo_ood", "covariate_transfer"],
        help="Split spec name.",
    )
    create.add_argument("--holdout-context", action="append", dest="holdout_contexts")
    create.add_argument("--holdout-perturbation", action="append", dest="holdout_perts")
    create.add_argument("--holdout-combo", action="append", dest="holdout_combos")
    create.add_argument("--covariate-name", dest="cov_name")
    create.add_argument("--holdout-covariate", action="append", dest="holdout_covs")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--val-frac", type=float, default=0.1)
    create.set_defaults(func=_cmd_splits_create)

    show = split_sub.add_parser("show", help="Show split summary by hash.")
    show.add_argument("hash", help="Split hash.")
    show.set_defaults(func=_cmd_splits_show)

    list_cmd = split_sub.add_parser("list", help="List stored splits.")
    list_cmd.set_defaults(func=_cmd_splits_list)

    import_pb = split_sub.add_parser("import-perturbench", help="Import official PerturBench splits into SplitStore.")
    import_pb.add_argument("--dataset", required=True, help="PerturBench dataset name or path.")
    import_pb.add_argument("--data", required=True, help="Path to dataset artifact (for index validation).")
    import_pb.add_argument("--split-dir", help="Optional directory containing split JSON files.")
    import_pb.set_defaults(func=_cmd_splits_import_perturbench)


def _add_train(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Training utilities.")
    train_sub = parser.add_subparsers(dest="train_command", required=True)

    baseline = train_sub.add_parser("baseline", help="Train and evaluate a baseline model.")
    baseline.add_argument("--data", required=True, help="Path to dataset artifact.")
    baseline.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    baseline.add_argument(
        "--baseline",
        required=True,
        choices=[
            "control_only",
            "global_mean",
            "per_perturbation_mean",
            "per_perturbation_context_mean",
            "additive_mean",
            "latent_shift",
            "ridge",
        ],
    )
    baseline.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization (ridge baseline only).")
    baseline.add_argument("--n-components", type=int, default=32, help="Latent shift PCA components (latent_shift only).")
    baseline.add_argument("--ensemble-size", type=int, default=1)
    baseline.add_argument("--conformal", action="store_true")
    baseline.add_argument("--out", default=None, help="Optional output run directory.")
    baseline.set_defaults(func=_cmd_train_baseline)

    v0 = train_sub.add_parser("perturbfm-v0", help="Train and evaluate PerturbFM v0.")
    v0.add_argument("--data", required=True, help="Path to dataset artifact.")
    v0.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v0.add_argument("--hidden-dim", type=int, default=128)
    v0.add_argument("--lr", type=float, default=1e-3)
    v0.add_argument("--epochs", type=int, default=50)
    v0.add_argument("--device", default="cpu")
    v0.add_argument("--batch-size", type=int, default=None)
    v0.add_argument("--seed", type=int, default=0)
    v0.add_argument("--pretrained-encoder", help="Path to pretrained CellEncoder checkpoint.")
    v0.add_argument("--freeze-encoder", action="store_true", help="Freeze pretrained CellEncoder weights.")
    v0.add_argument("--no-basal", action="store_true")
    v0.add_argument("--no-context", action="store_true")
    v0.add_argument("--no-perturbation", action="store_true")
    v0.add_argument("--ensemble-size", type=int, default=1)
    v0.add_argument("--conformal", action="store_true")
    v0.add_argument("--out", default=None, help="Optional output run directory.")
    v0.set_defaults(func=_cmd_train_perturbfm_v0)

    v1 = train_sub.add_parser("perturbfm-v1", help="Train and evaluate PerturbFM v1 (graph-augmented).")
    v1.add_argument("--data", required=True, help="Path to dataset artifact.")
    v1.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v1.add_argument("--adjacency", required=True, help="Path to .npz with key 'adjacency'.")
    v1.add_argument("--pert-masks", required=True, help="Path to JSON mapping pert_id -> list of gene indices.")
    v1.add_argument("--hidden-dim", type=int, default=128)
    v1.add_argument("--lr", type=float, default=1e-3)
    v1.add_argument("--epochs", type=int, default=50)
    v1.add_argument("--device", default="cpu")
    v1.add_argument("--batch-size", type=int, default=None)
    v1.add_argument("--seed", type=int, default=0)
    v1.add_argument("--pretrained-encoder", help="Path to pretrained CellEncoder checkpoint.")
    v1.add_argument("--freeze-encoder", action="store_true", help="Freeze pretrained CellEncoder weights.")
    v1.add_argument("--no-graph", action="store_true")
    v1.add_argument("--no-gating", action="store_true")
    v1.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v1.add_argument("--ensemble-size", type=int, default=1)
    v1.add_argument("--conformal", action="store_true")
    v1.add_argument("--out", default=None, help="Optional output run directory.")
    v1.set_defaults(func=_cmd_train_perturbfm_v1)

    v2 = train_sub.add_parser("perturbfm-v2", help="Train and evaluate PerturbFM v2 (CGIO).")
    v2.add_argument("--data", required=True, help="Path to dataset artifact.")
    v2.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v2.add_argument("--hidden-dim", type=int, default=128)
    v2.add_argument("--lr", type=float, default=1e-3)
    v2.add_argument("--epochs", type=int, default=50)
    v2.add_argument("--device", default="cpu")
    v2.add_argument("--batch-size", type=int, default=None)
    v2.add_argument("--seed", type=int, default=0)
    v2.add_argument("--no-gating", action="store_true")
    v2.add_argument("--no-contextual-operator", action="store_true")
    v2.add_argument("--num-bases", type=int, default=4)
    v2.add_argument("--adjacency", action="append", default=None, help="Path to .npz with key 'adjacency' (may be provided multiple times for multi-graph).")
    v2.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v2.add_argument("--ensemble-size", type=int, default=1)
    v2.add_argument("--conformal", action="store_true")
    v2.add_argument("--out", default=None, help="Optional output run directory.")
    v2.set_defaults(func=_cmd_train_perturbfm_v2)

    v2_res = train_sub.add_parser("perturbfm-v2-residual", help="Train and evaluate PerturbFM v2 residual (CGIO + additive).")
    v2_res.add_argument("--data", required=True, help="Path to dataset artifact.")
    v2_res.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v2_res.add_argument("--hidden-dim", type=int, default=128)
    v2_res.add_argument("--lr", type=float, default=1e-3)
    v2_res.add_argument("--epochs", type=int, default=50)
    v2_res.add_argument("--device", default="cpu")
    v2_res.add_argument("--batch-size", type=int, default=None)
    v2_res.add_argument("--seed", type=int, default=0)
    v2_res.add_argument("--no-gating", action="store_true")
    v2_res.add_argument("--no-contextual-operator", action="store_true")
    v2_res.add_argument("--num-bases", type=int, default=4)
    v2_res.add_argument("--adjacency", action="append", default=None, help="Path to .npz with key 'adjacency' (may be provided multiple times for multi-graph).")
    v2_res.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v2_res.add_argument("--combo-weight", type=float, default=1.0, help="Loss weight for multi-gene combo rows.")
    v2_res.add_argument("--ensemble-size", type=int, default=1)
    v2_res.add_argument("--conformal", action="store_true")
    v2_res.add_argument("--out", default=None, help="Optional output run directory.")
    v2_res.set_defaults(func=_cmd_train_perturbfm_v2_residual)

    v3 = train_sub.add_parser("perturbfm-v3", help="Train and evaluate PerturbFM v3 (state-dependent CGIO).")
    v3.add_argument("--data", required=True, help="Path to dataset artifact.")
    v3.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v3.add_argument("--hidden-dim", type=int, default=128)
    v3.add_argument("--lr", type=float, default=1e-3)
    v3.add_argument("--epochs", type=int, default=50)
    v3.add_argument("--device", default="cpu")
    v3.add_argument("--batch-size", type=int, default=None)
    v3.add_argument("--seed", type=int, default=0)
    v3.add_argument("--pretrained-encoder", help="Path to pretrained CellEncoder checkpoint.")
    v3.add_argument("--freeze-encoder", action="store_true", help="Freeze pretrained CellEncoder weights.")
    v3.add_argument("--no-gating", action="store_true")
    v3.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v3.add_argument("--adjacency", action="append", default=None, help="Path to .npz with key 'adjacency' (may be provided multiple times for multi-graph).")
    v3.add_argument("--ensemble-size", type=int, default=1)
    v3.add_argument("--conformal", action="store_true")
    v3.add_argument("--out", default=None, help="Optional output run directory.")
    v3.set_defaults(func=_cmd_train_perturbfm_v3)

    v3a = train_sub.add_parser("perturbfm-v3a", help="Train and evaluate PerturbFM v3a (Transformer cell encoder).")
    v3a.add_argument("--data", required=True, help="Path to dataset artifact.")
    v3a.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v3a.add_argument("--hidden-dim", type=int, default=128)
    v3a.add_argument("--lr", type=float, default=1e-3)
    v3a.add_argument("--epochs", type=int, default=50)
    v3a.add_argument("--device", default="cpu")
    v3a.add_argument("--batch-size", type=int, default=None)
    v3a.add_argument("--seed", type=int, default=0)
    v3a.add_argument("--no-gating", action="store_true")
    v3a.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v3a.add_argument("--adjacency", action="append", default=None, help="Path to .npz with key 'adjacency' (may be provided multiple times for multi-graph).")
    v3a.add_argument("--n-heads", type=int, default=4)
    v3a.add_argument("--n-layers", type=int, default=2)
    v3a.add_argument("--dropout", type=float, default=0.1)
    v3a.add_argument("--ensemble-size", type=int, default=1)
    v3a.add_argument("--conformal", action="store_true")
    v3a.add_argument("--out", default=None, help="Optional output run directory.")
    v3a.set_defaults(func=_cmd_train_perturbfm_v3a)

    v3_res = train_sub.add_parser("perturbfm-v3-residual", help="Train and evaluate PerturbFM v3 residual (state-dependent CGIO + additive).")
    v3_res.add_argument("--data", required=True, help="Path to dataset artifact.")
    v3_res.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v3_res.add_argument("--hidden-dim", type=int, default=128)
    v3_res.add_argument("--lr", type=float, default=1e-3)
    v3_res.add_argument("--epochs", type=int, default=50)
    v3_res.add_argument("--device", default="cpu")
    v3_res.add_argument("--batch-size", type=int, default=None)
    v3_res.add_argument("--seed", type=int, default=0)
    v3_res.add_argument("--pretrained-encoder", help="Path to pretrained CellEncoder checkpoint.")
    v3_res.add_argument("--freeze-encoder", action="store_true", help="Freeze pretrained CellEncoder weights.")
    v3_res.add_argument("--no-gating", action="store_true")
    v3_res.add_argument("--gating-mode", choices=["none", "scalar", "node", "lowrank", "mlp"], default=None)
    v3_res.add_argument("--adjacency", action="append", default=None, help="Path to .npz with key 'adjacency' (may be provided multiple times for multi-graph).")
    v3_res.add_argument("--combo-weight", type=float, default=1.0, help="Loss weight for multi-gene combo rows.")
    v3_res.add_argument("--ensemble-size", type=int, default=1)
    v3_res.add_argument("--conformal", action="store_true")
    v3_res.add_argument("--out", default=None, help="Optional output run directory.")
    v3_res.set_defaults(func=_cmd_train_perturbfm_v3_residual)


def _add_eval(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluation utilities.")
    eval_sub = parser.add_subparsers(dest="eval_command", required=True)

    preds = eval_sub.add_parser("predictions", help="Evaluate a predictions artifact.")
    preds.add_argument("--data", required=True, help="Path to dataset artifact.")
    preds.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    preds.add_argument("--preds", required=True, help="Path to predictions.npz.")
    preds.add_argument("--out", required=True, help="Output directory for metrics/report.")
    preds.set_defaults(func=_cmd_eval_predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perturbfm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_doctor(subparsers)
    _add_config(subparsers)
    _add_data(subparsers)
    _add_splits(subparsers)
    _add_train(subparsers)
    _add_eval(subparsers)
    return parser


def _cmd_doctor(_args: argparse.Namespace) -> int:
    print("perturbfm: ok")
    return 0


def _cmd_config(_args: argparse.Namespace) -> int:
    print("config: not implemented yet")
    return 0


def _cmd_data_make_synth(args: argparse.Namespace) -> int:
    from perturbfm.data.registry import make_synthetic_dataset

    ds = make_synthetic_dataset(
        n_obs=args.n_obs,
        n_genes=args.n_genes,
        n_contexts=args.n_contexts,
        n_perts=args.n_perts,
        seed=args.seed,
    )
    ds.save_artifact(args.out)
    print(f"Wrote synthetic dataset to {args.out}")
    print("Schema: data.npz keys [X_control, X_pert, delta, obs_idx]; meta.json keys [obs, var, metadata].")
    return 0


def _cmd_data_info(args: argparse.Namespace) -> int:
    from perturbfm.data.canonical import PerturbDataset

    ds = PerturbDataset.load_artifact(args.path)
    print(f"n_obs={ds.n_obs} n_genes={ds.n_genes}")
    print(f"obs fields={sorted(ds.obs.keys())}")
    return 0


def _cmd_data_import_perturbench(args: argparse.Namespace) -> int:
    from perturbfm.data.adapters.perturbench import PerturBenchAdapter

    adapter = PerturBenchAdapter(args.dataset)
    ds = adapter.load(backed=args.backed)
    ds.save_artifact(args.out)
    print(f"Wrote PerturBench artifact to {args.out}")
    print(f"n_obs={ds.n_obs} n_genes={ds.n_genes}")
    return 0


def _cmd_splits_create(args: argparse.Namespace) -> int:
    from perturbfm.data.canonical import PerturbDataset
    from perturbfm.data.splits.split_spec import (
        context_ood_split,
        perturbation_ood_split,
        combo_generalization_split,
        covariate_transfer_split,
    )
    from perturbfm.data.splits.split_store import SplitStore

    ds = PerturbDataset.load_artifact(args.data)
    if args.spec == "context_ood":
        if not args.holdout_contexts:
            raise ValueError("holdout contexts required")
        split = context_ood_split(
            ds.obs["context_id"],
            args.holdout_contexts,
            obs_perts=ds.obs["pert_id"],
            seed=args.seed,
            val_fraction=args.val_frac,
            require_shared_perturbations=True,
        )
    elif args.spec == "perturbation_ood":
        if not args.holdout_perts:
            raise ValueError("holdout perturbations required")
        split = perturbation_ood_split(ds.obs["pert_id"], args.holdout_perts, seed=args.seed, val_fraction=args.val_frac)
    elif args.spec == "combo_ood":
        if not args.holdout_combos:
            raise ValueError("holdout combos required")
        split = combo_generalization_split(ds.obs["pert_id"], args.holdout_combos, seed=args.seed, val_fraction=args.val_frac)
    elif args.spec == "covariate_transfer":
        if not args.cov_name or not args.holdout_covs:
            raise ValueError("covariate name and holdout covariates required")
        covs = ds.obs.get("covariates", {}).get(args.cov_name)
        if covs is None:
            raise ValueError(f"covariate {args.cov_name} not found in dataset.obs['covariates']")
        split = covariate_transfer_split(covs, args.holdout_covs, seed=args.seed, val_fraction=args.val_frac)
    else:
        raise ValueError(f"Unknown spec {args.spec}")
    split.freeze()
    store = SplitStore.default()
    store.save(split)
    print(f"split_hash={split.frozen_hash}")
    return 0


def _cmd_splits_show(args: argparse.Namespace) -> int:
    from perturbfm.data.splits.split_store import SplitStore

    store = SplitStore.default()
    split = store.load(args.hash)
    print(f"hash={split.frozen_hash}")
    print(f"train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")
    print(f"ood_axes={split.ood_axes}")
    return 0


def _cmd_splits_list(_args: argparse.Namespace) -> int:
    from perturbfm.data.splits.split_store import SplitStore

    store = SplitStore.default()
    hashes = store.list()
    if not hashes:
        print("No splits found.")
        return 0
    for split_hash in hashes:
        split = store.load(split_hash)
        print(
            f"{split_hash} train={len(split.train_idx)} val={len(split.val_idx)} "
            f"test={len(split.test_idx)} ood_axes={split.ood_axes}"
        )
    return 0


def _cmd_splits_import_perturbench(args: argparse.Namespace) -> int:
    from perturbfm.data.adapters.perturbench import PerturBenchAdapter
    from perturbfm.data.canonical import PerturbDataset
    from perturbfm.data.splits.split_store import SplitStore

    ds = PerturbDataset.load_artifact(args.data)
    adapter = PerturBenchAdapter(args.dataset)
    payloads = adapter.load_official_splits(args.split_dir)
    store = SplitStore.default()
    imported = 0
    for name, payload in payloads.items():
        split = adapter.parse_split_payload(payload, n_obs=ds.n_obs, name=name)
        split.freeze()
        store.save(split)
        imported += 1
    print(f"Imported {imported} PerturBench splits into {store.root}")
    return 0


def _cmd_train_baseline(args: argparse.Namespace) -> int:
    from perturbfm.eval.evaluator import run_baseline

    kwargs = {}
    if args.baseline == "ridge":
        kwargs["alpha"] = args.alpha
    if args.baseline == "latent_shift":
        kwargs["n_components"] = args.n_components
    run_dir = run_baseline(
        data_path=args.data,
        split_hash=args.split,
        baseline_name=args.baseline,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
        out_dir=args.out,
        **kwargs,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v0(args: argparse.Namespace) -> int:
    from perturbfm.eval.evaluator import run_perturbfm_v0

    run_dir = run_perturbfm_v0(
        data_path=args.data,
        split_hash=args.split,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        use_basal=not args.no_basal,
        use_context=not args.no_context,
        use_perturbation=not args.no_perturbation,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v1(args: argparse.Namespace) -> int:
    import json
    import numpy as np

    from perturbfm.eval.evaluator import run_perturbfm_v1

    with np.load(args.adjacency) as npz:
        adjacency = npz["adjacency"]
    pert_map = json.loads(Path(args.pert_masks).read_text(encoding="utf-8"))
    pert_gene_masks = {}
    for pert_id, indices in pert_map.items():
        mask = np.zeros(adjacency.shape[0], dtype=np.float32)
        for idx in indices:
            mask[int(idx)] = 1.0
        pert_gene_masks[pert_id] = mask

    run_dir = run_perturbfm_v1(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjacency,
        pert_gene_masks=pert_gene_masks,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        use_graph=not args.no_graph,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        combo_weight=args.combo_weight,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v2(args: argparse.Namespace) -> int:
    import numpy as np
    from perturbfm.eval.evaluator import run_perturbfm_v2

    adjs = None
    if args.adjacency:
        adjs = []
        for path in args.adjacency:
            with np.load(path) as npz:
                adjs.append(npz["adjacency"])
    run_dir = run_perturbfm_v2(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjs,
        pert_gene_masks=None,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        contextual_operator=not args.no_contextual_operator,
        num_bases=args.num_bases,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v2_residual(args: argparse.Namespace) -> int:
    import numpy as np
    from perturbfm.eval.evaluator import run_perturbfm_v2_residual

    adjs = None
    if args.adjacency:
        adjs = []
        for path in args.adjacency:
            with np.load(path) as npz:
                adjs.append(npz["adjacency"])
    run_dir = run_perturbfm_v2_residual(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjs,
        pert_gene_masks=None,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        contextual_operator=not args.no_contextual_operator,
        num_bases=args.num_bases,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v3(args: argparse.Namespace) -> int:
    import numpy as np
    from perturbfm.eval.evaluator import run_perturbfm_v3

    adjs = None
    if args.adjacency:
        adjs = []
        for path in args.adjacency:
            with np.load(path) as npz:
                adjs.append(npz["adjacency"])
    run_dir = run_perturbfm_v3(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjs,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        combo_weight=args.combo_weight,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v3a(args: argparse.Namespace) -> int:
    import numpy as np
    from perturbfm.eval.evaluator import run_perturbfm_v3a

    adjs = None
    if args.adjacency:
        adjs = []
        for path in args.adjacency:
            with np.load(path) as npz:
                adjs.append(npz["adjacency"])
    run_dir = run_perturbfm_v3a(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjs,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_train_perturbfm_v3_residual(args: argparse.Namespace) -> int:
    import numpy as np
    from perturbfm.eval.evaluator import run_perturbfm_v3_residual

    adjs = None
    if args.adjacency:
        adjs = []
        for path in args.adjacency:
            with np.load(path) as npz:
                adjs.append(npz["adjacency"])
    run_dir = run_perturbfm_v3_residual(
        data_path=args.data,
        split_hash=args.split,
        adjacency=adjs,
        out_dir=args.out,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        pretrained_encoder=args.pretrained_encoder,
        freeze_encoder=args.freeze_encoder,
        use_gating=not args.no_gating,
        gating_mode=args.gating_mode,
        ensemble_size=args.ensemble_size,
        conformal=args.conformal,
    )
    print(f"run_dir={run_dir}")
    return 0


def _cmd_eval_predictions(args: argparse.Namespace) -> int:
    from perturbfm.eval.evaluator import evaluate_predictions

    out_dir = evaluate_predictions(
        data_path=args.data,
        split_hash=args.split,
        predictions_path=args.preds,
        out_dir=args.out,
    )
    print(f"eval_dir={out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
