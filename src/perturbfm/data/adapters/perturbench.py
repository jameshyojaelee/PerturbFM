"""PerturBench adapter (local loading only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from perturbfm.data.canonical import PerturbDataset


class PerturBenchAdapter:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, **_kwargs: Any) -> PerturbDataset:
        return PerturbDataset.load_artifact(self.root)

    def load_official_splits(self, split_dir: Optional[str | Path] = None) -> Dict[str, Any]:
        raise NotImplementedError(
            "Official PerturBench split loading is not implemented yet. "
            "Place split artifacts locally and extend this adapter."
        )
