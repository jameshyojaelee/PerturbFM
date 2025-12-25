#!/usr/bin/env python3
"""Generate sweep configs for Norman19 doubles hyperparameter grid (Prompt 36)."""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import List

from perturbfm.utils.config import config_hash
from perturbfm.utils.hashing import stable_json_dumps


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
    ap.add_argument("--split", default="25e12ff5c7a6b67dc47c85d8ece8ac19bdd2d86141b403c8525972c654beb0c8")
    ap.add_argument("--data", default="data/artifacts/perturbench/norman19")
    ap.add_argument("--adjacency", default="graphs/norman19_identity.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", default="5,15,30")
    ap.add_argument("--hidden-dim", default="32,64,128")
    ap.add_argument("--lr", default="1e-3,3e-4")
    ap.add_argument("--batch-size", default="512,1024")
    args = ap.parse_args()

    out_dir = Path(args.out)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    epochs_list = [int(x) for x in args.epochs.split(",")]
    hidden_list = [int(x) for x in args.hidden_dim.split(",")]
    lr_list = [float(x) for x in args.lr.split(",")]
    batch_list = [int(x) for x in args.batch_size.split(",")]

    config_list = []
    manifest = []

    for epochs, hidden_dim, lr, batch_size in product(epochs_list, hidden_list, lr_list, batch_list):
        models = [
            {"kind": "baseline", "name": "control_only"},
            {"kind": "baseline", "name": "additive_mean"},
            _model_block("v2", epochs, hidden_dim, lr, batch_size, args.device, args.adjacency),
            _model_block("v2_residual", epochs, hidden_dim, lr, batch_size, args.device, args.adjacency),
            _model_block("v3", epochs, hidden_dim, lr, batch_size, args.device, args.adjacency),
            _model_block("v3_residual", epochs, hidden_dim, lr, batch_size, args.device, args.adjacency),
        ]
        cfg = {
            "datasets": [
                {
                    "name": "norman19_doubles_partial",
                    "path": args.data,
                    "splits": [args.split],
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
        manifest.append({"hash": h, "path": str(cfg_path), "epochs": epochs, "hidden_dim": hidden_dim, "lr": lr, "batch_size": batch_size})
        config_list.append(str(cfg_path))

    (out_dir / "manifest.json").write_text(stable_json_dumps(manifest), encoding="utf-8")
    list_path = out_dir / "config_list.txt"
    list_path.write_text("\n".join(config_list), encoding="utf-8")

    print(f"Wrote {len(config_list)} configs to {configs_dir}")
    print(f"Config list: {list_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
