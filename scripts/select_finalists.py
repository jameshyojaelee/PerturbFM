#!/usr/bin/env python3
"""Select top-K configs from validation summary and emit a test-run list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from perturbfm.utils.hashing import stable_json_dumps


def _metric(rec: Dict[str, Any], metric: str) -> float | None:
    panel, key = metric.split(".", 1)
    metrics = rec.get("metrics", {})
    if panel not in metrics:
        return None
    if panel == "uncertainty":
        return metrics[panel].get(key)
    return metrics[panel].get("global", {}).get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="runs_summary_val.json from suite runner.")
    ap.add_argument("--metric", default="perturbench.RMSE", help="panel.key for selection.")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out", required=True, help="Output JSON with finalist configs.")
    args = ap.parse_args()

    data = json.loads(Path(args.summary).read_text())
    runs = [r for r in data.get("runs", []) if r.get("status") == "ok"]

    # group by config_hash
    by_cfg: Dict[str, Dict[str, Any]] = {}
    for rec in runs:
        cfg_hash = rec["config_hash"]
        val = _metric(rec, args.metric)
        if val is None:
            continue
        entry = by_cfg.setdefault(cfg_hash, {"model": rec["model"], "data_path": rec["data_path"], "split_hash": rec["split_hash"], "values": []})
        entry["values"].append(float(val))

    scored = []
    for cfg_hash, entry in by_cfg.items():
        vals = entry["values"]
        mean = sum(vals) / len(vals)
        scored.append((mean, cfg_hash, entry))

    scored.sort(key=lambda x: x[0])
    finalists = scored[: args.top_k]

    payload = {
        "metric": args.metric,
        "top_k": args.top_k,
        "finalists": [
            {
                "config_hash": cfg_hash,
                "score_mean": mean,
                "model": entry["model"],
                "data_path": entry["data_path"],
                "split_hash": entry["split_hash"],
            }
            for mean, cfg_hash, entry in finalists
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stable_json_dumps(payload), encoding="utf-8")
    print(f"Wrote finalists to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
