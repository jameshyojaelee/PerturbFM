"""PerturBench adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from perturbfm.data.canonical import PerturbDataset, REQUIRED_OBS_FIELDS
from perturbfm.data.splits.split_spec import Split


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
        Path("third_party") / "PerturBench" / "datasets" / str(root),
        Path("third_party") / "PerturBench" / str(root),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"PerturBench dataset not found: {root_path}")


def _load_h5ad(path: Path, backed: bool = False) -> PerturbDataset:
    try:
        import anndata as ad  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "anndata is required to load PerturBench .h5ad files; install extras via `pip install -e \".[bench]\"`."
        ) from exc
    adata = ad.read_h5ad(path, backed="r" if backed else None)
    missing = [col for col in REQUIRED_OBS_FIELDS if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"PerturBench .h5ad missing required obs columns: {missing}")

    X_pert = _to_dense(adata.X)
    X_control = _to_dense(adata.layers["control"]) if "control" in adata.layers else None
    delta = _to_dense(adata.layers["delta"]) if "delta" in adata.layers else None
    if delta is None and X_control is not None:
        delta = np.asarray(X_pert) - np.asarray(X_control)
    obs = {k: adata.obs[k].tolist() for k in adata.obs.columns}
    var = adata.var_names.tolist()
    metadata = {"name": path.stem, "source": "perturbench"}
    return PerturbDataset(X_control=X_control, X_pert=X_pert, delta=delta, obs=obs, var=var, metadata=metadata)


class PerturBenchAdapter:
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
        raise FileNotFoundError(f"Unsupported PerturBench dataset format at {self.root}")

    def load_official_splits(self, split_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        split_dir = Path(split_dir) if split_dir else self._default_split_dir()
        if not split_dir.exists():
            raise FileNotFoundError("No official splits found; place them under <dataset>/splits or pass --split-dir.")
        splits = {}
        for fp in split_dir.glob("*.json"):
            splits[fp.stem] = json.loads(fp.read_text())
        return splits

    @staticmethod
    def parse_split_payload(payload: Dict[str, Any], n_obs: int, name: str | None = None) -> Split:
        def _extract(*keys: str) -> Any:
            for key in keys:
                if key in payload:
                    return payload[key]
            return None

        def _coerce_indices(value: Any, label: str) -> np.ndarray:
            if value is None:
                raise ValueError(f"Split payload missing {label} indices.")
            arr = np.asarray(value)
            if arr.dtype == bool:
                if arr.size != n_obs:
                    raise ValueError(f"Boolean mask length mismatch for {label}.")
                idx = np.where(arr)[0]
            else:
                idx = arr.astype(np.int64)
            if idx.min(initial=0) < 0 or idx.max(initial=-1) >= n_obs:
                raise ValueError(f"Split {label} indices out of bounds for n_obs={n_obs}.")
            return idx

        train_idx = _coerce_indices(_extract("train_idx", "train"), "train")
        val_idx = _coerce_indices(_extract("val_idx", "val"), "val")
        test_idx = _coerce_indices(_extract("test_idx", "test"), "test")
        calib_val = _extract("calib_idx", "calib")
        calib_idx = _coerce_indices(calib_val, "calib") if calib_val is not None else None
        notes = dict(payload.get("notes", {}))
        if name:
            notes.setdefault("name", name)
        notes.setdefault("source", "perturbench")
        split = Split(
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            calib_idx=calib_idx,
            ood_axes=payload.get("ood_axes", {}),
            notes=notes,
            seed=int(payload.get("seed", 0)),
        )
        return split
