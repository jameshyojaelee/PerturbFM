"""scPerturBench adapter stub (external-only)."""

from __future__ import annotations


class ScPerturBenchAdapter:
    def __init__(self, _root: str):
        raise RuntimeError(
            "scPerturBench is GPL-licensed and must remain external. "
            "Clone it into third_party/ and implement a local loader that "
            "maps into PerturbDataset without vendoring GPL code."
        )
