"""Tahoe-100M local parquet utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np


def _require_pyarrow():
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Tahoe parquet ingestion. Install with `pip install -e .[tahoe]`.") from exc


@dataclass
class TahoeConfig:
    root: Path
    data_glob: str = "data/train-*.parquet"
    batch_size: int = 1024
    shuffle_files: bool = False
    seed: int = 0


def list_shards(root: Path, pattern: str = "data/train-*.parquet") -> List[Path]:
    shards = sorted(root.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"No Tahoe shards found with pattern {pattern} under {root}")
    return shards


def iter_tahoe_batches(cfg: TahoeConfig) -> Iterator[Dict[str, np.ndarray]]:
    _require_pyarrow()
    import pyarrow.parquet as pq
    import numpy as np
    rng = np.random.default_rng(cfg.seed)

    shards = list_shards(cfg.root, cfg.data_glob)
    if cfg.shuffle_files:
        rng.shuffle(shards)

    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=cfg.batch_size):
            table = batch.to_pydict()
            genes = table["genes"]
            expr_list = table["expressions"]
            if not expr_list:
                continue
            ref_len = len(expr_list[0])
            keep = [i for i, row in enumerate(expr_list) if len(row) == ref_len]
            if not keep:
                continue
            expr = np.stack([np.asarray(expr_list[i], dtype=np.float32) for i in keep], axis=0)
            # metadata fields (strings) are left as Python lists
            def _subset_list(key):
                vals = table.get(key)
                if vals is None:
                    return None
                return [vals[i] for i in keep]

            yield {
                "genes": [genes[i] for i in keep],
                "expressions": expr,
                "drug": _subset_list("drug"),
                "sample": _subset_list("sample"),
                "cell_line_id": _subset_list("cell_line_id"),
                "moa-fine": _subset_list("moa-fine"),
                "canonical_smiles": _subset_list("canonical_smiles"),
            }


def load_metadata_table(root: Path, name: str) -> "np.ndarray":
    _require_pyarrow()
    import pyarrow.parquet as pq

    path = root / "metadata" / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata table: {path}")
    return pq.read_table(path).to_pandas()
