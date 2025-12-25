#!/usr/bin/env python3
"""Quick smoke test for prompt 35 configs (tiny subset)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import Split
from perturbfm.data.splits.split_store import SplitStore
from perturbfm.train.trainer import (
    fit_predict_baseline,
    fit_predict_perturbfm_v2,
    fit_predict_perturbfm_v2_residual,
    fit_predict_perturbfm_v3,
    fit_predict_perturbfm_v3_residual,
    get_baseline,
)


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix in (".yml", ".yaml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _subset_split(split, n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.asarray(split.train_idx, dtype=np.int64)[:n_train]
    test_idx = np.asarray(split.test_idx, dtype=np.int64)[:n_test]
    subset_idx = np.unique(np.concatenate([train_idx, test_idx]))
    remap = {int(orig): i for i, orig in enumerate(subset_idx)}
    train_small = np.array([remap[int(i)] for i in train_idx], dtype=np.int64)
    test_small = np.array([remap[int(i)] for i in test_idx], dtype=np.int64)
    return subset_idx, train_small, test_small


def _load_adjacencies(adj_cfg) -> List[np.ndarray] | None:
    if adj_cfg is None:
        return None
    adj_paths = adj_cfg if isinstance(adj_cfg, list) else [adj_cfg]
    adjs = []
    for path in adj_paths:
        with np.load(path) as npz:
            adjs.append(npz["adjacency"])
    return adjs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/suites/perturbench_norman19_doubles_report.json")
    ap.add_argument("--n-train", type=int, default=256)
    ap.add_argument("--n-test", type=int, default=128)
    ap.add_argument("--device", default=None, help="Override device for v2/v2_residual (cpu/cuda).")
    args = ap.parse_args()

    config = _load_config(Path(args.config))
    dataset_cfg = config["datasets"][0]
    data_path = Path(dataset_cfg["path"])
    split_hash = dataset_cfg["splits"][0]

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    store = SplitStore.default()
    split = store.load(split_hash)
    ds = PerturbDataset.load_artifact(data_path)

    subset_idx, train_small, test_small = _subset_split(split, args.n_train, args.n_test)
    ds_small = ds.select(subset_idx)
    split_small = Split(train_idx=train_small, val_idx=np.array([], dtype=np.int64), test_idx=test_small, calib_idx=np.array([], dtype=np.int64))

    print(f"Smoke subset: n_obs={ds_small.n_obs} train={len(train_small)} test={len(test_small)}")

    for model_cfg in config.get("models", []):
        kind = model_cfg.get("kind")
        if kind == "baseline":
            name = model_cfg["name"]
            model = get_baseline(name)
            out = fit_predict_baseline(model, ds_small, split_small)
            print(f"baseline:{name} ok mean={out['mean'].shape} var={out['var'].shape}")
            continue
        if kind in ("v2", "v2_residual", "v3", "v3_residual"):
            try:
                import torch
            except Exception as exc:
                raise RuntimeError("torch required for v2 smoke test") from exc
            device = args.device or model_cfg.get("device", "cpu")
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDA not available; falling back to cpu for smoke test.")
                device = "cpu"
            adj = _load_adjacencies(model_cfg.get("adjacency"))
            params = {
                "hidden_dim": int(model_cfg.get("hidden_dim", 16)),
                "lr": float(model_cfg.get("lr", 1e-3)),
                "epochs": 1,
                "device": device,
                "batch_size": min(64, int(model_cfg.get("batch_size", 64))),
                "seed": int(model_cfg.get("seed", 0)),
                "use_gating": not model_cfg.get("no_gating", False),
                "gating_mode": model_cfg.get("gating_mode"),
                "adjacencies": adj,
            }
            if kind in ("v2", "v2_residual"):
                params["contextual_operator"] = not model_cfg.get("no_contextual_operator", False)
                params["num_bases"] = int(model_cfg.get("num_bases", 2))
            if "combo_weight" in model_cfg:
                params["combo_weight"] = float(model_cfg["combo_weight"])
            if kind == "v2":
                out = fit_predict_perturbfm_v2(ds_small, split_small, **params)
            elif kind == "v2_residual":
                out = fit_predict_perturbfm_v2_residual(ds_small, split_small, **params)
            elif kind == "v3":
                out = fit_predict_perturbfm_v3(ds_small, split_small, **params)
            else:
                out = fit_predict_perturbfm_v3_residual(ds_small, split_small, **params)
            print(f"{kind} ok mean={out['mean'].shape} var={out['var'].shape}")
            continue
        print(f"Skipping unsupported kind: {kind}")

    print("Smoke test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
