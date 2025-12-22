"""Gene graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class GeneGraph:
    genes: List[str]
    adjacency: torch.Tensor  # shape [G, G]

    @staticmethod
    def from_edge_list(edges: Sequence[Tuple[str, str]], genes: Sequence[str]) -> "GeneGraph":
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        g = len(genes)
        adj = np.zeros((g, g), dtype=np.float32)
        for a, b in edges:
            if a in gene_to_idx and b in gene_to_idx:
                i, j = gene_to_idx[a], gene_to_idx[b]
                adj[i, j] = 1.0
                adj[j, i] = 1.0
        adj = _normalize_adj(adj)
        return GeneGraph(genes=list(genes), adjacency=torch.as_tensor(adj))

    def subset(self, genes_subset: Sequence[str]) -> "GeneGraph":
        idx = [self.genes.index(g) for g in genes_subset if g in self.genes]
        adj = self.adjacency[idx][:, idx]
        return GeneGraph(genes=list(genes_subset), adjacency=adj)


def _normalize_adj(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1, keepdims=True) + 1e-6
    return adj / deg
