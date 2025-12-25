#!/usr/bin/env python3
"""Generate multiple partial-double splits and configs (Prompt 39)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import combo_generalization_split
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.utils.config import config_hash
from perturbfm.utils.hashing import stable_json_dumps


def _normalize_genes(genes) -> List[str]:
    if genes is None:
        return []
    if isinstance(genes, str):
        return [genes]
    if isinstance(genes, np.ndarray):
        genes = genes.tolist()
    return [str(g) for g in genes]


def _combo_gene_map(pert_ids: Sequence[str], pert_genes: Sequence[Sequence[str]]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for pid, genes in zip(pert_ids, pert_genes):
        genes_norm = _normalize_genes(genes)
        if len(genes_norm) < 2:
            continue
        key = str(pid)
        mapping.setdefault(key, set()).update(genes_norm)
    return mapping


def _select_partial_combos(combo_ids: Sequence[str], combo_gene_map: dict[str, set[str]], seed: int, frac: float) -> List[str]:
    combos_sorted = sorted(set(combo_ids))
    holdout_count = max(1, int(round(len(combos_sorted) * frac)))
    holdout_count = min(holdout_count, len(combos_sorted))

    all_genes = sorted({g for genes in combo_gene_map.values() for g in genes})
    selected: set[str] = set()
    remaining_genes = set(all_genes)

    while remaining_genes and len(selected) < holdout_count:
        best_combo = None
        best_gain = -1
        for combo in combos_sorted:
            if combo in selected:
                continue
            gain = len(combo_gene_map.get(combo, set()) & remaining_genes)
            if gain > best_gain:
                best_gain = gain
                best_combo = combo
        if best_combo is None or best_gain <= 0:
            break
        selected.add(best_combo)
        remaining_genes -= combo_gene_map.get(best_combo, set())

    rng = np.random.default_rng(seed)
    remaining = [c for c in combos_sorted if c not in selected]
    if len(selected) < holdout_count and remaining:
        n_more = min(holdout_count - len(selected), len(remaining))
        extra = rng.choice(remaining, size=n_more, replace=False)
        selected.update([str(c) for c in extra])
    return sorted(selected)


def _model_block(kind: str, epochs: int, hidden_dim: int, lr: float, batch_size: int, device: str, adjacency: str):
    return {
        "kind": kind,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "lr": lr,
        "batch_size": batch_size,
        "device": device,
        "adjacency": adjacency,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output sweep directory.")
    ap.add_argument("--data", default="data/artifacts/perturbench/norman19")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adjacency", default="graphs/norman19_identity.npz")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--fractions", default="0.1,0.2,0.3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    ds = PerturbDataset.load_artifact(args.data)
    pert_ids = [str(p) for p in ds.obs["pert_id"]]
    pert_genes = ds.obs.get("pert_genes", [[] for _ in range(ds.n_obs)])
    combo_ids = sorted({pid for pid, genes in zip(pert_ids, pert_genes) if len(_normalize_genes(genes)) >= 2})
    combo_gene_map = _combo_gene_map(pert_ids, pert_genes)

    store = SplitStore(root=Path(args.split_dir)) if args.split_dir else SplitStore.default()

    config_list = []
    manifest = []
    for frac in [float(x) for x in args.fractions.split(",")]:
        holdout = _select_partial_combos(combo_ids, combo_gene_map, seed=args.seed, frac=frac)
        split = combo_generalization_split(pert_ids, holdout_combos=holdout, seed=args.seed, val_fraction=0.1)
        split.notes["split_type"] = "combo_partial_doubles"
        split.notes["partial_frac"] = frac
        split.freeze()
        try:
            store.save(split)
        except ValueError:
            split = store.load(split.frozen_hash)

        models = [
            {"kind": "baseline", "name": "control_only"},
            {"kind": "baseline", "name": "additive_mean"},
            _model_block("v2", args.epochs, args.hidden_dim, args.lr, args.batch_size, args.device, args.adjacency),
            _model_block("v2_residual", args.epochs, args.hidden_dim, args.lr, args.batch_size, args.device, args.adjacency),
            _model_block("v3", args.epochs, args.hidden_dim, args.lr, args.batch_size, args.device, args.adjacency),
            _model_block("v3_residual", args.epochs, args.hidden_dim, args.lr, args.batch_size, args.device, args.adjacency),
        ]
        cfg = {
            "datasets": [
                {
                    "name": f"norman19_doubles_partial_frac{frac}",
                    "path": args.data,
                    "splits": [split.frozen_hash],
                }
            ],
            "models": models,
            "seeds": [0, 1, 2],
            "score_metric": "perturbench.RMSE",
            "score_mode": "min",
        }
        h = config_hash(cfg)
        cfg_path = configs_dir / f"{h}.json"
        cfg_path.write_text(stable_json_dumps(cfg), encoding="utf-8")
        manifest.append({"hash": h, "path": str(cfg_path), "partial_frac": frac, "split_hash": split.frozen_hash})
        config_list.append(str(cfg_path))

    (out_dir / "manifest.json").write_text(stable_json_dumps(manifest), encoding="utf-8")
    list_path = out_dir / "config_list.txt"
    list_path.write_text("\n".join(config_list), encoding="utf-8")

    print(f"Wrote {len(config_list)} configs to {configs_dir}")
    print(f"Config list: {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
