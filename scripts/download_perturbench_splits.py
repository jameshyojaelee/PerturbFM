#!/usr/bin/env python3
"""
Download PerturBench split JSONs from the Hugging Face dataset repo.

Usage:
  python scripts/download_perturbench_splits.py --out /path/to/splits --contains split
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

from download_utils import cpu_workers, download_many


HF_API = "https://huggingface.co/api/datasets/altoslabs/perturbench"
HF_RESOLVE = "https://huggingface.co/datasets/altoslabs/perturbench/resolve/main/"


def _list_json_files(contains: list[str]) -> list[str]:
    with urllib.request.urlopen(HF_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    siblings = payload.get("siblings", [])
    files = [item.get("rfilename") for item in siblings if item.get("rfilename")]
    json_files = [f for f in files if f.endswith(".json")]
    if not contains:
        return json_files
    keep = []
    for fname in json_files:
        if any(tok in fname for tok in contains):
            keep.append(fname)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for downloaded split JSONs.")
    ap.add_argument("--contains", action="append", default=["split"], help="Substring filter for JSON filenames.")
    ap.add_argument("--max-workers", type=int, default=None, help="Parallel download workers.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_json_files(args.contains)
    if not files:
        raise RuntimeError("No JSON files matched the filter; adjust --contains.")

    items = []
    for fname in files:
        url = HF_RESOLVE + urllib.parse.quote(fname)
        items.append((url, out_dir / Path(fname).name))

    workers = args.max_workers or min(16, cpu_workers())
    download_many(items, max_workers=workers)
    print(f"Downloaded {len(items)} JSON files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
