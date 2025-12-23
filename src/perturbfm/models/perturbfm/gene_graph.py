"""Gene graph utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch


@dataclass
class GeneGraph:
    genes: List[str]
    edge_index: torch.Tensor  # shape [2, E]
    edge_weight: torch.Tensor  # shape [E]
    num_nodes: int

    @staticmethod
    def from_edge_list(edges: Sequence[Tuple[str, str]], genes: Sequence[str]) -> "GeneGraph":
        gene_to_idx = {g: i for i, g in enumerate(genes)}
        g = len(genes)
        edges = []
        for a, b in edges:
            if a in gene_to_idx and b in gene_to_idx:
                i, j = gene_to_idx[a], gene_to_idx[b]
                edges.append((i, j))
                edges.append((j, i))
        if not edges:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_weight = torch.zeros((0,), dtype=torch.float32)
            return GeneGraph(genes=list(genes), edge_index=edge_index, edge_weight=edge_weight, num_nodes=g)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        edge_weight = _normalize_edge_weights(edge_index, edge_weight, g)
        return GeneGraph(genes=list(genes), edge_index=edge_index, edge_weight=edge_weight, num_nodes=g)

    def subset(self, genes_subset: Sequence[str]) -> "GeneGraph":
        idx = [self.genes.index(g) for g in genes_subset if g in self.genes]
        idx_map = {old: new for new, old in enumerate(idx)}
        mask = torch.isin(self.edge_index[0], torch.tensor(idx)) & torch.isin(self.edge_index[1], torch.tensor(idx))
        edge_index = self.edge_index[:, mask]
        if edge_index.numel() == 0:
            return GeneGraph(genes=list(genes_subset), edge_index=edge_index, edge_weight=torch.zeros((0,), dtype=torch.float32), num_nodes=len(genes_subset))
        edge_index = torch.stack([torch.tensor([idx_map[int(i)] for i in edge_index[0]]),
                                  torch.tensor([idx_map[int(i)] for i in edge_index[1]])], dim=0)
        edge_weight = self.edge_weight[mask]
        edge_weight = _normalize_edge_weights(edge_index, edge_weight, len(genes_subset))
        return GeneGraph(genes=list(genes_subset), edge_index=edge_index, edge_weight=edge_weight, num_nodes=len(genes_subset))


def _normalize_edge_weights(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # row-normalize by source node
    src = edge_index[0]
    deg = torch.zeros((num_nodes,), dtype=edge_weight.dtype)
    deg.index_add_(0, src, edge_weight)
    deg = deg + 1e-6
    return edge_weight / deg[src]
