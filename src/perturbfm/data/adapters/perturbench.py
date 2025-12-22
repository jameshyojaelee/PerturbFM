"""PerturBench adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset


def _load_h5ad(path: Path) -> PerturbDataset:
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:
        raise RuntimeError("anndata is required to load PerturBench .h5ad files; install extras [bench].") from exc
    adata = ad.read_h5ad(path)
    X_pert = np.asarray(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    X_control = np.asarray(adata.layers["control"]) if "control" in adata.layers else None
    delta = np.asarray(adata.layers["delta"]) if "delta" in adata.layers else None
    obs = {k: adata.obs[k].tolist() for k in adata.obs.columns}
    var = adata.var_names.tolist()
    metadata = {"name": path.stem}
    return PerturbDataset(X_control=X_control, X_pert=X_pert, delta=delta, obs=obs, var=var, metadata=metadata)


class PerturBenchAdapter:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, **_kwargs: Any) -> PerturbDataset:
        if self.root.is_dir() and (self.root / "data.npz").exists():
            return PerturbDataset.load_artifact(self.root)
        if self.root.suffix == ".h5ad":
            return _load_h5ad(self.root)
        raise FileNotFoundError(f"Unsupported PerturBench dataset format at {self.root}")

    def load_official_splits(self, split_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        split_dir = Path(split_dir) if split_dir else self.root / "splits"
        if not split_dir.exists():
            raise FileNotFoundError("No official splits found; place them under <dataset>/splits or pass --split-dir.")
        splits = {}
        for fp in split_dir.glob("*.json"):
            splits[fp.stem] = json.loads(fp.read_text())
        return splits
