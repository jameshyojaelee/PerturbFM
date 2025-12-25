#!/usr/bin/env python3
"""Aggregate sweep array outputs into a leaderboard."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from perturbfm.utils.hashing import stable_json_dumps


def _mean_std(values: List[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(statistics.mean(values)), float(statistics.pstdev(values))


def _metric(rec: Dict[str, Any], panel: str, key: str) -> float | None:
    metrics = rec.get("metrics", {})
    if panel not in metrics:
        return None
    if panel == "uncertainty":
        return metrics[panel].get(key)
    return metrics[panel].get("global", {}).get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Sweep root containing task_* directories.")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--metric", default="perturbench.RMSE")
    args = ap.parse_args()

    panel, key = args.metric.split(".", 1)
    root = Path(args.root)
    tasks = sorted(p for p in root.glob("task_*") if p.is_dir())

    rows = []
    for task in tasks:
        summary_path = task / "runs_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        runs = [r for r in summary.get("runs", []) if r.get("status") == "ok"]
        if not runs:
            continue

        config_path = task / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}

        by_model: Dict[str, List[float]] = {}
        for rec in runs:
            label = rec["model_label"]
            val = _metric(rec, panel, key)
            if val is None:
                continue
            by_model.setdefault(label, []).append(float(val))

        additive_vals = by_model.get("baseline:additive_mean", [])
        add_mean, _ = _mean_std(additive_vals)

        model_stats = {}
        for label, vals in by_model.items():
            mean, std = _mean_std(vals)
            ratio = mean / add_mean if add_mean and label != "baseline:additive_mean" else (1.0 if label == "baseline:additive_mean" else float("nan"))
            model_stats[label] = {"mean": mean, "std": std, "ratio_vs_additive": ratio}

        rows.append(
            {
                "task": task.name,
                "config": config,
                "models": model_stats,
            }
        )

    out_json = Path(args.out_json)
    out_json.write_text(stable_json_dumps({"metric": args.metric, "tasks": rows}), encoding="utf-8")

    # markdown
    out_md = Path(args.out_md)
    lines = [f"# Sweep leaderboard ({args.metric})", ""]
    for row in rows:
        lines.append(f"## {row['task']}")
        lines.append("")
        lines.append("| model | mean ± std | ratio vs additive |")
        lines.append("|---|---:|---:|")
        for label, stats in sorted(row["models"].items()):
            mean = stats["mean"]
            std = stats["std"]
            ratio = stats["ratio_vs_additive"]
            lines.append(f"| {label} | {mean:.6f} ± {std:.6f} | {ratio:.6f} |")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
