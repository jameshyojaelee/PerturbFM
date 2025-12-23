"""Split store for immutable, hash-locked splits."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from perturbfm.data.splits.split_spec import Split
from perturbfm.utils.hashing import stable_json_dumps


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


@dataclass
class SplitStore:
    root: Path

    @classmethod
    def default(cls) -> "SplitStore":
        override = os.environ.get("PERTURBFM_SPLIT_DIR")
        root = Path(override) if override else (_find_repo_root() / "splits")
        return cls(root=root)

    def _path_for(self, split_hash: str) -> Path:
        return self.root / f"{split_hash}.json"

    def save(self, split: Split) -> Path:
        split.assert_frozen()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(split.frozen_hash)
        payload = split.to_dict()
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if stable_json_dumps(existing) != stable_json_dumps(payload):
                raise ValueError(f"Split hash collision with different content: {split.frozen_hash}")
            return path
        path.write_text(stable_json_dumps(payload), encoding="utf-8")
        return path

    def list(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.json"))

    def load(self, split_hash: str) -> Split:
        path = self._path_for(split_hash)
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {split_hash}")
        payload: Dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        split = Split.from_dict(payload)
        computed = split.compute_hash()
        if payload.get("frozen_hash") != computed:
            raise ValueError("Split hash mismatch on load.")
        split.freeze()
        return split
