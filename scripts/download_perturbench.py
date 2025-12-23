#!/usr/bin/env python3
"""
Download all processed PerturBench .h5ad(.gz) files from Hugging Face.

Usage:
  python scripts/download_perturbench.py --out /path/to/perturbench_h5ad --max-workers 32
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


def _list_files() -> list[str]:
    with urllib.request.urlopen(HF_API, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    siblings = payload.get("siblings", [])
    files = [item.get("rfilename") for item in siblings if item.get("rfilename")]
    return [f for f in files if f.endswith(".h5ad") or f.endswith(".h5ad.gz")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for downloaded files.")
    ap.add_argument("--max-workers", type=int, default=None, help="Parallel download workers.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_files()
    if not files:
        raise RuntimeError("No .h5ad files found from Hugging Face API.")

    items = []
    for fname in files:
        url = HF_RESOLVE + urllib.parse.quote(fname)
        items.append((url, out_dir / Path(fname).name))

    workers = args.max_workers or min(32, cpu_workers())
    download_many(items, max_workers=workers)
    print(f"Downloaded {len(items)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
