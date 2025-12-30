#!/usr/bin/env python3
"""
Run a PerturBench-style suite and aggregate a scorecard.

Config format (JSON/YAML):
{
  "datasets": [
    {"name": "pb_dataset", "path": "/path/to/artifact", "splits": ["<SPLIT_HASH>"]}
  ],
  "splits": ["<SPLIT_HASH>"],  # optional global list
  "models": [
    {"kind": "baseline", "name": "global_mean"},
    {"kind": "v0", "epochs": 20}
  ],
  "seeds": [0, 1],
  "score_metric": "perturbench.RMSE",
  "score_mode": "min"
}
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from perturbfm.data.registry import make_synthetic_dataset
from perturbfm.data.splits.split_spec import context_ood_split
from perturbfm.data.splits.split_store import SplitStore
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
from perturbfm.utils.config import config_hash
from perturbfm.utils.graph_io import load_graph_npz
from perturbfm.utils.hashing import stable_json_dumps
from perturbfm.utils.seeds import set_seed


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix in (".yml", ".yaml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _model_label(cfg: Dict[str, Any]) -> str:
    kind = cfg.get("kind", "unknown")
    if kind == "baseline":
        return f"baseline:{cfg.get('name', 'unknown')}"
    return str(kind)


def _run_id(dataset: str, split_hash: str, model_label: str, seed: int, cfg_hash: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe_ds = dataset.replace("/", "_")
    safe_model = model_label.replace("/", "_").replace(":", "_")
    return f"{ts}_{safe_ds}_{split_hash[:7]}_{safe_model}_seed{seed}_{cfg_hash}"


def _run_model(cfg: Dict[str, Any], data_path: str, split_hash: str, run_dir: Path | None, eval_split: str) -> Path:
    kind = cfg.get("kind")
    if kind == "baseline":
        name = cfg["name"]
        params = {k: v for k, v in cfg.items() if k not in ("kind", "name")}
        return run_baseline(
            data_path,
            split_hash,
            baseline_name=name,
            out_dir=str(run_dir) if run_dir else None,
            eval_split=eval_split,
            **params,
        )
    if kind == "v0":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        return run_perturbfm_v0(data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params)
    if kind == "v1":
        import numpy as np
        import json as pyjson

        if "adjacency" not in cfg or "pert_masks" not in cfg:
            raise ValueError("v1 requires 'adjacency' and 'pert_masks' in config.")
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
        return run_perturbfm_v1(
            data_path,
            split_hash,
            adjacency=adjacency,
            pert_gene_masks=pert_gene_masks,
            out_dir=str(run_dir) if run_dir else None,
            eval_split=eval_split,
            **params,
        )
    if kind == "v2":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            import numpy as np

            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            adjs = []
            for path in adj_paths:
                adjs.append(load_graph_npz(path))
            params["adjacency"] = adjs
        return run_perturbfm_v2(data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params)
    if kind == "v2_residual":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            import numpy as np

            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            adjs = []
            for path in adj_paths:
                adjs.append(load_graph_npz(path))
            params["adjacency"] = adjs
        return run_perturbfm_v2_residual(
            data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params
        )
    if kind == "v3a":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            import numpy as np

            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            adjs = []
            for path in adj_paths:
                adjs.append(load_graph_npz(path))
            params["adjacency"] = adjs
        return run_perturbfm_v3a(data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params)
    if kind == "v3":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            import numpy as np

            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            adjs = []
            for path in adj_paths:
                adjs.append(load_graph_npz(path))
            params["adjacency"] = adjs
        return run_perturbfm_v3(data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params)
    if kind == "v3_residual":
        params = {k: v for k, v in cfg.items() if k != "kind"}
        if "adjacency" in params and params["adjacency"] is not None:
            import numpy as np

            adj_paths = params["adjacency"]
            if not isinstance(adj_paths, list):
                adj_paths = [adj_paths]
            adjs = []
            for path in adj_paths:
                adjs.append(load_graph_npz(path))
            params["adjacency"] = adjs
        return run_perturbfm_v3_residual(
            data_path, split_hash, out_dir=str(run_dir) if run_dir else None, eval_split=eval_split, **params
        )
    raise ValueError(f"Unknown model kind: {kind}")


def _iter_runs(config: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("Config must include at least one dataset.")
    global_splits = config.get("splits", [])
    models = config.get("models", [])
    if not models:
        raise ValueError("Config must include at least one model config.")
    seeds = config.get("seeds", [0])

    for ds in datasets:
        ds_path = ds["path"]
        ds_name = ds.get("name", Path(ds_path).name)
        splits = ds.get("splits", global_splits)
        if not splits:
            raise ValueError(f"Dataset {ds_name} has no splits configured.")
        for split_hash in splits:
            for model in models:
                for seed in seeds:
                    yield {
                        "dataset_name": ds_name,
                        "data_path": ds_path,
                        "split_hash": split_hash,
                        "model": model,
                        "seed": seed,
                    }


def _dry_run_config(work_dir: Path) -> Dict[str, Any]:
    data_dir = work_dir / "data"
    ds = make_synthetic_dataset(n_obs=60, n_genes=12, n_contexts=3, n_perts=4, seed=0)
    ds.save_artifact(data_dir)

    split = context_ood_split(ds.obs["context_id"], ["C0"], obs_perts=ds.obs["pert_id"], seed=0, val_fraction=0.2)
    split.freeze()
    store = SplitStore(root=work_dir / "splits")
    store.save(split)
    os.environ["PERTURBFM_SPLIT_DIR"] = str(store.root)

    return {
        "datasets": [{"name": "synthetic", "path": str(data_dir), "splits": [split.frozen_hash]}],
        "models": [{"kind": "baseline", "name": "global_mean"}],
        "seeds": [0],
        "score_metric": "perturbench.RMSE",
        "score_mode": "min",
    }


def _extract_metric(metrics: Dict[str, Any], metric: str) -> float | None:
    panel, key = metric.split(".", 1)
    panel_obj = metrics.get(panel, {})
    if isinstance(panel_obj, dict):
        global_section = panel_obj.get("global", {})
        return global_section.get(key)
    return None


def _aggregate_scorecard(records: List[Dict[str, Any]], metric: str, mode: str) -> Dict[str, Any]:
    tasks: Dict[str, Dict[str, List[float]]] = {}
    for rec in records:
        if rec["status"] != "ok":
            continue
        value = _extract_metric(rec["metrics"], metric)
        if value is None or np.isnan(value):
            continue
        task_key = f"{rec['dataset_name']}::{rec['split_hash']}"
        tasks.setdefault(task_key, {}).setdefault(rec["model_label"], []).append(float(value))

    task_scores = []
    for task_key, model_scores in tasks.items():
        scores = {m: float(np.mean(v)) for m, v in model_scores.items()}
        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=(mode == "max"))
        dataset_name, split_hash = task_key.split("::", 1)
        task_scores.append(
            {
                "dataset": dataset_name,
                "split_hash": split_hash,
                "scores": scores,
                "ranking": ranking,
            }
        )

    overall_scores: Dict[str, List[float]] = {}
    for task in task_scores:
        for model, score in task["scores"].items():
            overall_scores.setdefault(model, []).append(score)
    overall = {m: float(np.mean(v)) for m, v in overall_scores.items()}
    overall_rank = sorted(overall.items(), key=lambda kv: kv[1], reverse=(mode == "max"))

    return {
        "primary_metric": metric,
        "score_mode": mode,
        "tasks": task_scores,
        "overall": {"scores": overall, "ranking": overall_rank},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="Suite config JSON/YAML.")
    ap.add_argument("--out", required=True, help="Scorecard JSON path.")
    ap.add_argument("--run-root", help="Optional root for run directories.")
    ap.add_argument("--split-dir", help="Optional split store root override.")
    ap.add_argument("--dry-run", action="store_true", help="Run a tiny synthetic suite.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop on first failure.")
    ap.add_argument("--score-metric", help="Override config score metric (panel.key).")
    ap.add_argument("--score-mode", choices=["min", "max"], help="Override config score mode.")
    ap.add_argument("--eval-split", choices=["test", "val"], default="test", help="Evaluate on split.test_idx or split.val_idx.")
    args = ap.parse_args()

    if args.split_dir:
        os.environ["PERTURBFM_SPLIT_DIR"] = args.split_dir

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        config = _dry_run_config(out_path.parent / "dry_run")
    else:
        if not args.config:
            raise ValueError("--config is required unless --dry-run is set.")
        config = _load_config(Path(args.config))

    records = []
    git_commit = _git_commit()
    cmd = " ".join(sys.argv)
    run_root = Path(args.run_root) if args.run_root else None

    for run_cfg in _iter_runs(config):
        set_seed(int(run_cfg["seed"]))
        model_label = _model_label(run_cfg["model"])
        cfg_payload = {
            "dataset": run_cfg["dataset_name"],
            "data_path": run_cfg["data_path"],
            "split_hash": run_cfg["split_hash"],
            "model": run_cfg["model"],
            "seed": run_cfg["seed"],
        }
        cfg_hash = config_hash(cfg_payload)
        run_dir = None
        if run_root:
            run_dir = run_root / _run_id(run_cfg["dataset_name"], run_cfg["split_hash"], model_label, run_cfg["seed"], cfg_hash)
        record = {
            "dataset_name": run_cfg["dataset_name"],
            "data_path": run_cfg["data_path"],
            "split_hash": run_cfg["split_hash"],
            "model": run_cfg["model"],
            "model_label": model_label,
            "seed": run_cfg["seed"],
            "config_hash": cfg_hash,
            "git_commit": git_commit,
            "command": cmd,
            "eval_split": args.eval_split,
        }
        try:
            run_path = _run_model(run_cfg["model"], run_cfg["data_path"], run_cfg["split_hash"], run_dir, args.eval_split)
            metrics_path = Path(run_path) / "metrics.json"
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            record.update({"status": "ok", "run_dir": str(run_path), "metrics": metrics})
        except Exception as exc:
            record.update({"status": "failed", "error": str(exc)})
            if args.fail_fast:
                raise
        finally:
            try:
                import gc

                gc.collect()
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        records.append(record)

    summary_path = out_path.parent / f"runs_summary_{args.eval_split}.json"
    summary_payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "runs": records}
    summary_path.write_text(stable_json_dumps(summary_payload), encoding="utf-8")
    if args.eval_split == "test":
        legacy_path = out_path.parent / "runs_summary.json"
        legacy_path.write_text(stable_json_dumps(summary_payload), encoding="utf-8")

    metric = args.score_metric or config.get("score_metric", "perturbench.RMSE")
    mode = args.score_mode or config.get("score_mode", "min")
    scorecard = _aggregate_scorecard(records, metric, mode)
    scorecard["generated_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(stable_json_dumps(scorecard), encoding="utf-8")

    text_path = out_path.parent / "scorecard.txt"
    lines = [f"Primary metric: {metric} ({mode})", ""]
    for task in scorecard["tasks"]:
        lines.append(f"{task['dataset']}::{task['split_hash']}")
        for model, score in task["ranking"]:
            lines.append(f"  {model}: {score:.6f}")
        lines.append("")
    lines.append("Overall")
    for model, score in scorecard["overall"]["ranking"]:
        lines.append(f"  {model}: {score:.6f}")
    text_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote scorecard to {out_path}")
    print(f"Wrote run summary to {summary_path}")
    print(f"Wrote text summary to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
