#!/usr/bin/env python3
"""Create a Tahoe-100M manifest with shard hashes for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from perturbfm.utils.hashing import stable_json_dumps


def _hash_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Tahoe-100M root (contains data/ + metadata/).")
    ap.add_argument("--out", required=True, help="Output manifest JSON.")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of shards hashed (0 = all).")
    args = ap.parse_args()

    root = Path(args.root)
    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")

    shards = sorted(data_dir.glob("train-*.parquet"))
    if args.limit and args.limit > 0:
        shards = shards[: args.limit]

    manifest = []
    for path in shards:
        manifest.append({"path": str(path), "sha256": _hash_file(path)})

    payload = {
        "root": str(root),
        "shard_count": len(shards),
        "shards": manifest,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(stable_json_dumps(payload), encoding="utf-8")
    print(f"Wrote manifest with {len(shards)} shards to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
