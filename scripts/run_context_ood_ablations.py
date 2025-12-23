#!/usr/bin/env python3
"""
Run a small context-OOD ablation sweep using the suite runner.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from perturbfm.utils.hashing import stable_json_dumps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset artifact path.")
    ap.add_argument("--split", required=True, help="Context-OOD split hash.")
    ap.add_argument("--out", required=True, help="Scorecard JSON path.")
    ap.add_argument("--run-root", help="Optional run root directory.")
    args = ap.parse_args()

    config = {
        "datasets": [{"name": Path(args.data).name, "path": args.data, "splits": [args.split]}],
        "models": [
            {"kind": "baseline", "name": "control_only"},
            {"kind": "baseline", "name": "global_mean"},
            {"kind": "baseline", "name": "latent_shift", "n_components": 32},
            {"kind": "v0", "use_context": True},
            {"kind": "v0", "use_context": False},
        ],
        "seeds": [0],
        "score_metric": "scperturbench.MSE",
        "score_mode": "min",
    }

    cfg_path = Path(args.out).with_suffix(".config.json")
    cfg_path.write_text(stable_json_dumps(config), encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/run_perturbench_suite.py",
        "--config",
        str(cfg_path),
        "--out",
        str(args.out),
        "--score-metric",
        "scperturbench.MSE",
        "--score-mode",
        "min",
    ]
    if args.run_root:
        cmd.extend(["--run-root", args.run_root])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
