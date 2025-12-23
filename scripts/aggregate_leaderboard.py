#!/usr/bin/env python3
"""
Aggregate runs into a leaderboard, selecting by validation metrics only.

Usage:
  python scripts/aggregate_leaderboard.py --runs runs --out leaderboard.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from perturbfm.utils.hashing import stable_json_dumps


def _extract_metric(metrics: Dict[str, Any], metric: str) -> float | None:
    panel, key = metric.split(".", 1)
    panel_obj = metrics.get(panel, {})
    if isinstance(panel_obj, dict):
        global_section = panel_obj.get("global", {})
        return global_section.get(key)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs", help="Runs root directory.")
    ap.add_argument("--out", required=True, help="Leaderboard JSON path.")
    ap.add_argument("--metric", default="perturbench.RMSE", help="Metric panel.key (lower is better by default).")
    ap.add_argument("--mode", choices=["min", "max"], default="min")
    ap.add_argument("--val-metrics", default="val_metrics.json", help="Validation metrics filename to use for selection.")
    args = ap.parse_args()

    runs_root = Path(args.runs)
    records: List[Dict[str, Any]] = []
    for run_dir in runs_root.glob("*"):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.json"
        val_path = run_dir / args.val_metrics
        if not cfg_path.exists() or not val_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        val_metrics = json.loads(val_path.read_text())
        score = _extract_metric(val_metrics, args.metric)
        if score is None:
            continue
        record = {
            "run_dir": str(run_dir),
            "data_path": cfg.get("data_path"),
            "split_hash": cfg.get("split_hash"),
            "model": cfg.get("model", {}),
            "score": score,
        }
        records.append(record)

    leaderboard: Dict[str, Any] = {
        "selection_rule": f"Select best by validation metric {args.metric} ({args.mode}); test metrics are ignored.",
        "metric": args.metric,
        "mode": args.mode,
        "tasks": [],
    }

    tasks: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        task_key = f"{Path(rec['data_path']).name}::{rec['split_hash']}"
        tasks.setdefault(task_key, []).append(rec)

    for task_key, runs in tasks.items():
        runs_sorted = sorted(runs, key=lambda r: r["score"], reverse=(args.mode == "max"))
        best = runs_sorted[0]
        leaderboard["tasks"].append(
            {
                "task": task_key,
                "best_run": best,
                "all_runs": runs_sorted,
            }
        )

    out_path = Path(args.out)
    out_path.write_text(stable_json_dumps(leaderboard), encoding="utf-8")

    md_path = out_path.with_suffix(".md")
    lines = ["# Leaderboard", "", f"Selection rule: {leaderboard['selection_rule']}", ""]
    for task in leaderboard["tasks"]:
        lines.append(f"## {task['task']}")
        best = task["best_run"]
        lines.append(f"- best: {best['model'].get('name', 'unknown')} ({best['score']:.6f})")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote leaderboard to {out_path}")
    print(f"Wrote markdown summary to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
