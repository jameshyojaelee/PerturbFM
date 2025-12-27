#!/usr/bin/env python3
"""Run test evaluation for a finalists JSON (from select_finalists.py)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perturbfm.eval.evaluator import (
    run_baseline,
    run_perturbfm_v0,
    run_perturbfm_v1,
    run_perturbfm_v2,
    run_perturbfm_v2_residual,
    run_perturbfm_v3,
    run_perturbfm_v3_residual,
    run_perturbfm_v3a,
)
from perturbfm.utils.graph_io import load_graph_npz


def _run_model(cfg, data_path, split_hash, out_dir):
    kind = cfg.get("kind")
    if kind == "baseline":
        return run_baseline(data_path, split_hash, baseline_name=cfg["name"], out_dir=out_dir, eval_split="test", **{k: v for k, v in cfg.items() if k not in ("kind", "name")})
    if kind == "v0":
        return run_perturbfm_v0(data_path, split_hash, out_dir=out_dir, eval_split="test", **{k: v for k, v in cfg.items() if k != "kind"})
    if kind == "v1":
        import numpy as np
        import json as pyjson

        adjacency = load_graph_npz(cfg["adjacency"])
        if isinstance(adjacency, dict):
            raise ValueError("v1 requires dense adjacency with key 'adjacency' in the .npz file.")
        pert_map = pyjson.loads(Path(cfg["pert_masks"]).read_text(encoding="utf-8"))
        pert_gene_masks = {}
        for pert_id, indices in pert_map.items():
            mask = np.zeros(adjacency.shape[0], dtype=np.float32)
            for idx in indices:
                mask[int(idx)] = 1.0
            pert_gene_masks[pert_id] = mask
        params = {k: v for k, v in cfg.items() if k not in ("kind", "adjacency", "pert_masks")}
        return run_perturbfm_v1(data_path, split_hash, adjacency=adjacency, pert_gene_masks=pert_gene_masks, out_dir=out_dir, eval_split="test", **params)
    if kind == "v2":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            params["adjacency"] = [load_graph_npz(p) for p in adj_paths]
        return run_perturbfm_v2(data_path, split_hash, out_dir=out_dir, eval_split="test", **params)
    if kind == "v2_residual":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            params["adjacency"] = [load_graph_npz(p) for p in adj_paths]
        return run_perturbfm_v2_residual(data_path, split_hash, out_dir=out_dir, eval_split="test", **params)
    if kind == "v3":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            params["adjacency"] = [load_graph_npz(p) for p in adj_paths]
        return run_perturbfm_v3(data_path, split_hash, out_dir=out_dir, eval_split="test", **params)
    if kind == "v3_residual":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            params["adjacency"] = [load_graph_npz(p) for p in adj_paths]
        return run_perturbfm_v3_residual(data_path, split_hash, out_dir=out_dir, eval_split="test", **params)
    if kind == "v3a":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            params["adjacency"] = [load_graph_npz(p) for p in adj_paths]
        return run_perturbfm_v3a(data_path, split_hash, out_dir=out_dir, eval_split="test", **params)
    raise ValueError(f"Unknown model kind: {kind}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalists", required=True, help="Path to finalists JSON (from select_finalists.py).")
    ap.add_argument("--out-root", required=True, help="Root directory for test runs.")
    args = ap.parse_args()

    finalists = json.loads(Path(args.finalists).read_text())
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for i, entry in enumerate(finalists["finalists"]):
        cfg = entry["model"]
        data_path = entry["data_path"]
        split_hash = entry["split_hash"]
        out_dir = out_root / f"finalist_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_model(cfg, data_path, split_hash, str(out_dir))
        print(f"Completed finalist {i}: {cfg.get('kind', cfg.get('name'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
