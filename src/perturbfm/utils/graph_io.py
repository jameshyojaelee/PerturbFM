"""Graph file loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


def load_graph_npz(path: str | Path) -> np.ndarray | Dict[str, Any]:
    """Load a graph from an .npz file.

    Supports either:
    - dense adjacency: key "adjacency"
    - sparse edges: keys "edge_index" (+ optional "edge_weight", "genes", "num_nodes")
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as npz:
        files = set(npz.files)
        if "adjacency" in files:
            return np.asarray(npz["adjacency"], dtype=np.float32)
        if "edge_index" in files:
            edge_index = np.asarray(npz["edge_index"], dtype=np.int64)
            if edge_index.ndim != 2:
                raise ValueError(f"edge_index must be rank-2 in {path}")
            if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
                edge_index = edge_index.T
            if edge_index.shape[0] != 2:
                raise ValueError(f"edge_index must have shape [2, E] in {path}")
            if "edge_weight" in files:
                edge_weight = np.asarray(npz["edge_weight"], dtype=np.float32)
            else:
                edge_weight = np.ones(edge_index.shape[1], dtype=np.float32)
            if "num_nodes" in files:
                num_nodes = int(np.asarray(npz["num_nodes"]).item())
            elif "genes" in files:
                num_nodes = int(len(npz["genes"]))
            else:
                num_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
            return {"edge_index": edge_index, "edge_weight": edge_weight, "num_nodes": num_nodes}
    raise KeyError(f"Expected 'adjacency' or 'edge_index' in {path}")
