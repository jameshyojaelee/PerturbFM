#!/usr/bin/env python3
"""
Download processed scPerturb .h5ad files from Zenodo.

Usage:
  python scripts/download_scperturb.py --out /path/to/scperturb_h5ad --max-workers 32
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from download_utils import cpu_workers, download_many


ZENODO_RECORD = "https://zenodo.org/api/records/13350497"


def _list_files() -> list[tuple[str, str]]:
    with urllib.request.urlopen(ZENODO_RECORD, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files", [])
    out = []
    for entry in files:
        key = entry.get("key")
        links = entry.get("links", {})
        url = links.get("download") or links.get("self")
        if not key or not url:
            continue
        if key.endswith(".h5ad") or key.endswith(".h5ad.gz"):
            out.append((key, url))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory for downloaded files.")
    ap.add_argument("--max-workers", type=int, default=None, help="Parallel download workers.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _list_files()
    if not files:
        raise RuntimeError("No .h5ad files found in Zenodo record.")

    items = [(url, out_dir / Path(key).name) for key, url in files]
    workers = args.max_workers or min(32, cpu_workers())
    download_many(items, max_workers=workers)
    print(f"Downloaded {len(items)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
