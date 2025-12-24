#!/usr/bin/env python3
"""Create Norman19 double-perturbation splits (all doubles + partial holdout)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import combo_generalization_split
from perturbfm.data.splits.split_store import SplitStore


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


def _select_partial_combos(
    combo_ids: Sequence[str],
    combo_gene_map: dict[str, set[str]],
    seed: int,
    frac: float,
    count: int | None,
) -> List[str]:
    combos_sorted = sorted(set(combo_ids))
    if not combos_sorted:
        return []
    if count is None:
        holdout_count = max(1, int(round(len(combos_sorted) * frac)))
    else:
        holdout_count = max(1, int(count))
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


def _combo_fraction(indices: Iterable[int], pert_genes: Sequence[Sequence[str]]) -> float:
    idx = list(indices)
    if not idx:
        return 0.0
    combos = 0
    for i in idx:
        if len(_normalize_genes(pert_genes[i])) >= 2:
            combos += 1
    return combos / len(idx)


def _summarize_split(name: str, split, pert_genes: Sequence[Sequence[str]]) -> None:
    print(f"[{name}] hash={split.frozen_hash}")
    print(f"[{name}] train={len(split.train_idx)} val={len(split.val_idx)} test={len(split.test_idx)}")
    frac = _combo_fraction(split.test_idx, pert_genes)
    print(f"[{name}] test_combo_fraction={frac:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/artifacts/perturbench/norman19", help="Path to Norman19 artifact.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--partial-frac", type=float, default=0.2)
    ap.add_argument("--partial-count", type=int, default=None)
    ap.add_argument("--split-dir", default=None, help="Optional split store root.")
    args = ap.parse_args()

    data_path = Path(args.data)
    ds = PerturbDataset.load_artifact(data_path)
    pert_ids = [str(p) for p in ds.obs["pert_id"]]
    pert_genes = ds.obs.get("pert_genes", [[] for _ in range(ds.n_obs)])
    if pert_genes is None:
        raise ValueError("Dataset missing obs['pert_genes'].")

    combo_mask = np.array([len(_normalize_genes(g)) >= 2 for g in pert_genes], dtype=bool)
    combo_ids = sorted({pid for pid, is_combo in zip(pert_ids, combo_mask) if is_combo})
    if not combo_ids:
        raise ValueError("No combo perturbations found (len(pert_genes) >= 2).")

    plus_mask = np.array(["+" in pid for pid in pert_ids], dtype=bool)
    mismatch = int(np.sum(combo_mask != plus_mask))
    print(f"[dataset] combos={len(combo_ids)} rows={int(combo_mask.sum())} plus_mismatch_rows={mismatch}")

    combo_gene_map = _combo_gene_map(pert_ids, pert_genes)
    partial_holdout = _select_partial_combos(combo_ids, combo_gene_map, seed=args.seed, frac=args.partial_frac, count=args.partial_count)
    print(f"[partial] holdout_combos={len(partial_holdout)}")

    split_all = combo_generalization_split(pert_ids, holdout_combos=combo_ids, seed=args.seed, val_fraction=args.val_frac)
    split_all.notes["split_type"] = "combo_all_doubles"
    split_all.notes["combo_count"] = len(combo_ids)
    split_all.freeze()

    split_partial = combo_generalization_split(pert_ids, holdout_combos=partial_holdout, seed=args.seed, val_fraction=args.val_frac)
    split_partial.notes["split_type"] = "combo_partial_doubles"
    split_partial.notes["combo_count"] = len(partial_holdout)
    split_partial.notes["partial_frac"] = args.partial_frac
    split_partial.freeze()

    store = SplitStore(root=Path(args.split_dir)) if args.split_dir else SplitStore.default()
    store.save(split_all)
    store.save(split_partial)

    _summarize_split("all_doubles", split_all, pert_genes)
    _summarize_split("partial_doubles", split_partial, pert_genes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
