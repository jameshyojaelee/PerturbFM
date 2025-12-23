"""Canonical dataset abstraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from perturbfm.utils.hashing import stable_json_dumps

REQUIRED_OBS_FIELDS = ("pert_id", "context_id", "batch_id", "is_control")


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return list(value)


def _slice_array(arr: Any, idx: Sequence[int]) -> Any:
    if isinstance(arr, np.ndarray):
        return arr[idx]
    return [_as_list(arr)[i] for i in idx]


def _slice_obs(obs: Dict[str, Any], idx: Sequence[int]) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            sliced[key] = {k: _slice_array(v, idx) for k, v in value.items()}
        else:
            sliced[key] = _slice_array(value, idx)
    return sliced


@dataclass
class PerturbDataset:
    X_control: Optional[np.ndarray]
    X_pert: Optional[np.ndarray]
    delta: Optional[np.ndarray]
    obs: Dict[str, Any]
    var: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        for arr in (self.X_pert, self.X_control, self.delta):
            if arr is not None:
                return int(arr.shape[0])
        if self.obs:
            first = next(iter(self.obs.values()))
            return len(first)
        return 0

    @property
    def n_genes(self) -> int:
        for arr in (self.X_pert, self.X_control, self.delta):
            if arr is not None:
                return int(arr.shape[1])
        return len(self.var)

    def validate(self) -> None:
        n_obs = self.n_obs
        n_genes = self.n_genes

        for arr in (self.X_pert, self.X_control, self.delta):
            if arr is None:
                continue
            if arr.ndim != 2:
                raise ValueError("Expression arrays must be 2D [N, G].")
            if arr.shape[0] != n_obs or arr.shape[1] != n_genes:
                raise ValueError("Expression array shapes are inconsistent.")

        for field in REQUIRED_OBS_FIELDS:
            if field not in self.obs:
                raise ValueError(f"obs missing required field: {field}")

        for key, value in self.obs.items():
            if isinstance(value, dict):
                for subval in value.values():
                    if len(subval) != n_obs:
                        raise ValueError(f"obs.{key} length mismatch")
            else:
                if len(value) != n_obs:
                    raise ValueError(f"obs.{key} length mismatch")

        if self.X_control is not None and self.X_pert is not None and self.delta is not None:
            computed = self.X_pert - self.X_control
            if not np.allclose(self.delta, computed):
                raise ValueError("delta does not match X_pert - X_control.")

    def select(self, idx: Sequence[int]) -> "PerturbDataset":
        idx = list(idx)
        return PerturbDataset(
            X_control=_slice_array(self.X_control, idx) if self.X_control is not None else None,
            X_pert=_slice_array(self.X_pert, idx) if self.X_pert is not None else None,
            delta=_slice_array(self.delta, idx) if self.delta is not None else None,
            obs=_slice_obs(self.obs, idx),
            var=list(self.var),
            metadata=dict(self.metadata),
        )

    def to(self, device: str) -> "PerturbDataset":
        try:
            import torch
        except Exception as exc:  # pragma: no cover - torch absent
            raise RuntimeError("torch is required for PerturbDataset.to()") from exc

        def _to(arr: Optional[np.ndarray]) -> Optional["torch.Tensor"]:
            if arr is None:
                return None
            tensor = torch.as_tensor(arr)
            return tensor.to(device)

        return PerturbDataset(
            X_control=_to(self.X_control),
            X_pert=_to(self.X_pert),
            delta=_to(self.delta),
            obs=dict(self.obs),
            var=list(self.var),
            metadata=dict(self.metadata),
        )

    def save_artifact(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        data_path = out_path / "data.npz"
        meta_path = out_path / "meta.json"

        payload = {}
        if self.X_control is not None:
            payload["X_control"] = np.asarray(self.X_control, dtype=np.float32)
        if self.X_pert is not None:
            payload["X_pert"] = np.asarray(self.X_pert, dtype=np.float32)
        if self.delta is not None:
            payload["delta"] = np.asarray(self.delta, dtype=np.float32)
        payload["obs_idx"] = np.arange(self.n_obs, dtype=np.int64)
        np.savez_compressed(data_path, **payload)

        meta = {
            "schema_version": 1,
            "obs": {k: _as_list(v) if not isinstance(v, dict) else {kk: _as_list(vv) for kk, vv in v.items()} for k, v in self.obs.items()},
            "var": list(self.var),
            "metadata": self.metadata,
        }
        meta_path.write_text(stable_json_dumps(meta), encoding="utf-8")

    def save_memmap_artifact(self, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        memmap_dir = out_path / "memmap"
        memmap_dir.mkdir(parents=True, exist_ok=True)

        def _save(name: str, arr: Optional[np.ndarray]) -> None:
            if arr is None:
                return
            np.save(memmap_dir / f"{name}.npy", np.asarray(arr, dtype=np.float32))

        _save("X_control", self.X_control)
        _save("X_pert", self.X_pert)
        _save("delta", self.delta)
        np.save(memmap_dir / "obs_idx.npy", np.arange(self.n_obs, dtype=np.int64))

        meta = {
            "schema_version": 1,
            "obs": {k: _as_list(v) if not isinstance(v, dict) else {kk: _as_list(vv) for kk, vv in v.items()} for k, v in self.obs.items()},
            "var": list(self.var),
            "metadata": self.metadata,
            "artifact_type": "memmap",
        }
        (out_path / "meta.json").write_text(stable_json_dumps(meta), encoding="utf-8")

    @staticmethod
    def load_artifact(in_dir: str | Path) -> "PerturbDataset":
        in_path = Path(in_dir)
        data_path = in_path / "data.npz"
        meta_path = in_path / "meta.json"
        memmap_dir = in_path / "memmap"
        if not meta_path.exists():
            raise FileNotFoundError("Expected meta.json in dataset artifact.")
        if not data_path.exists() and not memmap_dir.exists():
            raise FileNotFoundError("Expected data.npz or memmap/ in dataset artifact.")

        if data_path.exists():
            with np.load(data_path) as npz:
                X_control = npz["X_control"] if "X_control" in npz else None
                X_pert = npz["X_pert"] if "X_pert" in npz else None
                delta = npz["delta"] if "delta" in npz else None
        else:
            def _load(name: str) -> Optional[np.ndarray]:
                path = memmap_dir / f"{name}.npy"
                if not path.exists():
                    return None
                return np.load(path, mmap_mode="r")

            X_control = _load("X_control")
            X_pert = _load("X_pert")
            delta = _load("delta")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obs = meta.get("obs", {})
        var = meta.get("var", [])
        metadata = meta.get("metadata", {})

        ds = PerturbDataset(
            X_control=X_control,
            X_pert=X_pert,
            delta=delta,
            obs=obs,
            var=var,
            metadata=metadata,
        )
        ds.validate()
        return ds
