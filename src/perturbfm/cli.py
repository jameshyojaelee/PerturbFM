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


def _add_splits(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("splits", help="Split utilities.")
    split_sub = parser.add_subparsers(dest="split_command", required=True)

    create = split_sub.add_parser("create", help="Create and store a split.")
    create.add_argument("--data", required=True, help="Path to dataset artifact directory.")
    create.add_argument("--spec", required=True, choices=["context_ood"], help="Split spec name.")
    create.add_argument("--holdout-context", action="append", required=True, dest="holdout_contexts")
    create.add_argument("--seed", type=int, default=0)
    create.add_argument("--val-frac", type=float, default=0.1)
    create.set_defaults(func=_cmd_splits_create)

    show = split_sub.add_parser("show", help="Show split summary by hash.")
    show.add_argument("hash", help="Split hash.")
    show.set_defaults(func=_cmd_splits_show)


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
            "global_mean",
            "per_perturbation_mean",
            "per_perturbation_context_mean",
            "ridge",
        ],
    )
    baseline.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization (ridge baseline only).")
    baseline.add_argument("--out", default=None, help="Optional output run directory.")
    baseline.set_defaults(func=_cmd_train_baseline)

    v0 = train_sub.add_parser("perturbfm-v0", help="Train and evaluate PerturbFM v0.")
    v0.add_argument("--data", required=True, help="Path to dataset artifact.")
    v0.add_argument("--split", required=True, help="Split hash (must exist in split store).")
    v0.add_argument("--hidden-dim", type=int, default=128)
    v0.add_argument("--lr", type=float, default=1e-3)
    v0.add_argument("--epochs", type=int, default=50)
    v0.add_argument("--device", default="cpu")
    v0.add_argument("--no-basal", action="store_true")
    v0.add_argument("--no-context", action="store_true")
    v0.add_argument("--no-perturbation", action="store_true")
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
    v1.add_argument("--no-graph", action="store_true")
    v1.add_argument("--no-gating", action="store_true")
    v1.add_argument("--out", default=None, help="Optional output run directory.")
    v1.set_defaults(func=_cmd_train_perturbfm_v1)


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


def _cmd_splits_create(args: argparse.Namespace) -> int:
    from perturbfm.data.canonical import PerturbDataset
    from perturbfm.data.splits.split_spec import context_ood_split
    from perturbfm.data.splits.split_store import SplitStore

    ds = PerturbDataset.load_artifact(args.data)
    split = context_ood_split(ds.obs["context_id"], args.holdout_contexts, seed=args.seed, val_fraction=args.val_frac)
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


def _cmd_train_baseline(args: argparse.Namespace) -> int:
    from perturbfm.eval.evaluator import run_baseline

    kwargs = {}
    if args.baseline == "ridge":
        kwargs["alpha"] = args.alpha
    run_dir = run_baseline(
        data_path=args.data,
        split_hash=args.split,
        baseline_name=args.baseline,
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
        use_basal=not args.no_basal,
        use_context=not args.no_context,
        use_perturbation=not args.no_perturbation,
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
        use_graph=not args.no_graph,
        use_gating=not args.no_gating,
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
