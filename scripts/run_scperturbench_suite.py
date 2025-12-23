#!/usr/bin/env python3
"""
Wrapper for running a scPerturBench-style suite with scPerturBench defaults.
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    cmd = [
        sys.executable,
        "scripts/run_perturbench_suite.py",
        "--score-metric",
        "scperturbench.MSE",
        "--score-mode",
        "min",
    ]
    cmd.extend(sys.argv[1:])
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
