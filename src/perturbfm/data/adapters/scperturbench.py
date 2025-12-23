"""scPerturBench adapter (external-only data access)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset, REQUIRED_OBS_FIELDS


def _to_dense(array: Any) -> np.ndarray:
    if array is None:
        return None  # type: ignore[return-value]
    if hasattr(array, "toarray"):
        return np.asarray(array.toarray())
    if hasattr(array, "todense"):
        return np.asarray(array.todense())
    return np.asarray(array)


def _resolve_root(root: str | Path) -> Path:
    root_path = Path(root)
    if root_path.exists():
        return root_path
    candidates = [
        Path("third_party") / "scPerturBench" / "datasets" / str(root),
        Path("third_party") / "scPerturBench" / str(root),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"scPerturBench dataset not found: {root_path}")


def _load_h5ad(path: Path, backed: bool = False) -> PerturbDataset:
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "anndata is required to load scPerturBench .h5ad files; install extras via `pip install -e \".[bench]\"`."
        ) from exc
    adata = ad.read_h5ad(path, backed="r" if backed else None)
    missing = [col for col in REQUIRED_OBS_FIELDS if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"scPerturBench .h5ad missing required obs columns: {missing}")

    X_pert = _to_dense(adata.X)
    X_control = _to_dense(adata.layers["control"]) if "control" in adata.layers else None
    delta = _to_dense(adata.layers["delta"]) if "delta" in adata.layers else None
    if delta is None and X_control is not None:
        delta = np.asarray(X_pert) - np.asarray(X_control)
    obs = {k: adata.obs[k].tolist() for k in adata.obs.columns}
    var = adata.var_names.tolist()
    metadata = {"name": path.stem, "source": "scperturbench"}
    return PerturbDataset(X_control=X_control, X_pert=X_pert, delta=delta, obs=obs, var=var, metadata=metadata)


class ScPerturBenchAdapter:
    def __init__(self, root: str | Path):
        self.root = _resolve_root(root)

    def _default_split_dir(self) -> Path:
        if self.root.is_file():
            return self.root.parent / "splits"
        return self.root / "splits"

    def load(self, **kwargs: Any) -> PerturbDataset:
        backed = bool(kwargs.get("backed", False))
        if self.root.is_dir() and (self.root / "data.npz").exists():
            return PerturbDataset.load_artifact(self.root)
        if self.root.suffix == ".h5ad":
            return _load_h5ad(self.root, backed=backed)
        raise FileNotFoundError(f"Unsupported scPerturBench dataset format at {self.root}")

    def load_official_splits(self, split_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        split_dir = Path(split_dir) if split_dir else self._default_split_dir()
        if not split_dir.exists():
            raise FileNotFoundError("No scPerturBench splits found; place them under <dataset>/splits or pass --split-dir.")
        splits = {}
        for fp in split_dir.glob("*.json"):
            splits[fp.stem] = json.loads(fp.read_text())
        return splits
