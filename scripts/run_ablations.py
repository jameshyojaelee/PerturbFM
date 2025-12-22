#!/usr/bin/env python3
"""
Run a set of ablations and summarize results.

Usage:
  python scripts/run_ablations.py --data <artifact_dir> --split <hash> --configs configs.json --out runs_summary.json

configs.json example:
[
  {"kind": "baseline", "name": "global_mean"},
  {"kind": "baseline", "name": "per_perturbation_mean"},
  {"kind": "v0"},
  {"kind": "v1", "use_graph": true, "use_gating": true},
  {"kind": "v2", "use_gating": true, "contextual_operator": true}
]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perturbfm.eval.evaluator import (
    run_baseline,
    run_perturbfm_v0,
    run_perturbfm_v1,
    run_perturbfm_v2,
)
from perturbfm.utils.hashing import stable_json_dumps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--configs", required=True, help="Path to JSON list of config objects.")
    ap.add_argument("--out", required=True, help="Summary JSON path.")
    args = ap.parse_args()

    configs = json.loads(Path(args.configs).read_text())
    results = []
    for cfg in configs:
        kind = cfg.get("kind")
        if kind == "baseline":
            run_dir = run_baseline(args.data, args.split, baseline_name=cfg["name"], **{k: v for k, v in cfg.items() if k not in ("kind", "name")})
        elif kind == "v0":
            run_dir = run_perturbfm_v0(args.data, args.split, **{k: v for k, v in cfg.items() if k != "kind"})
        elif kind == "v1":
            run_dir = run_perturbfm_v1(args.data, args.split, **{k: v for k, v in cfg.items() if k != "kind"})
        elif kind == "v2":
            run_dir = run_perturbfm_v2(args.data, args.split, adjacency=None, **{k: v for k, v in cfg.items() if k != "kind"})
        else:
            raise ValueError(f"Unknown kind {kind}")
        metrics_path = Path(run_dir) / "metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        results.append({"config": cfg, "run_dir": str(run_dir), "metrics": metrics})

    Path(args.out).write_text(stable_json_dumps(results), encoding="utf-8")
    print(f"Wrote summary to {args.out}")


if __name__ == "__main__":
    main()
