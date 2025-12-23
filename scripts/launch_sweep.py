#!/usr/bin/env python3
"""
Launch a sweep by expanding a simple grid and writing per-run configs.

Usage:
  python scripts/launch_sweep.py --base configs/base.json --grid configs/grid.json --out sweeps/run1
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from perturbfm.utils.config import config_hash
from perturbfm.utils.hashing import stable_json_dumps


def _load_config(path: Path) -> Dict[str, Any]:
    if path.suffix in (".yml", ".yaml"):
        import yaml

        return yaml.safe_load(path.read_text())
    return json.loads(path.read_text())


def _set_path(cfg: Dict[str, Any], path: str, value: Any) -> None:
    tokens = []
    for part in path.split("."):
        if "[" in part and part.endswith("]"):
            name, idx = part[:-1].split("[")
            tokens.append(name)
            tokens.append(int(idx))
        else:
            tokens.append(part)
    cur: Any = cfg
    for token in tokens[:-1]:
        if isinstance(token, int):
            cur = cur[token]
        else:
            cur = cur.setdefault(token, {})
    last = tokens[-1]
    if isinstance(last, int):
        cur[last] = value
    else:
        cur[last] = value


def _expand_grid(base: Dict[str, Any], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = deepcopy(base)
        for key, val in zip(keys, combo):
            _set_path(cfg, key, val)
        configs.append(cfg)
    return configs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base suite config (JSON/YAML).")
    ap.add_argument("--grid", required=True, help="Grid spec JSON/YAML (dict of key -> list).")
    ap.add_argument("--out", required=True, help="Output sweep directory.")
    ap.add_argument("--submit", action="store_true", help="Submit SLURM array via scripts/slurm/sweep_array.sbatch.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    base = _load_config(Path(args.base))
    grid = _load_config(Path(args.grid))
    configs = _expand_grid(base, grid)

    manifest = []
    config_list = []
    for cfg in configs:
        h = config_hash(cfg)
        cfg_path = configs_dir / f"{h}.json"
        cfg_path.write_text(stable_json_dumps(cfg), encoding="utf-8")
        manifest.append({"hash": h, "path": str(cfg_path)})
        config_list.append(str(cfg_path))

    (out_dir / "manifest.json").write_text(stable_json_dumps(manifest), encoding="utf-8")
    list_path = out_dir / "config_list.txt"
    list_path.write_text("\n".join(config_list), encoding="utf-8")

    print(f"Wrote {len(configs)} configs to {configs_dir}")
    print(f"Manifest: {out_dir / 'manifest.json'}")
    print(f"Config list: {list_path}")

    if args.submit:
        array_range = f"0-{len(configs) - 1}"
        cmd = [
            "sbatch",
            f"--array={array_range}",
            f"--export=ALL,CONFIG_LIST={list_path},OUT_ROOT={out_dir}",
            "scripts/slurm/sweep_array.sbatch",
        ]
        print("Submitting:", " ".join(cmd))
        subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
