"""Perturbation encoder for PerturbFM."""

from __future__ import annotations

import torch
from torch import nn


class PerturbationEncoder(nn.Module):
    def __init__(self, num_perts: int, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(num_perts, hidden_dim)

    def forward(self, pert_idx: torch.Tensor) -> torch.Tensor:
        return self.embed(pert_idx)


class GraphPerturbationEncoder(nn.Module):
    def __init__(
        self,
        adjacency: torch.Tensor,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_graph: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.use_gating = use_gating
        self.num_layers = num_layers
        self.register_buffer("adjacency", adjacency)
        if use_gating:
            self.edge_gates = nn.Parameter(torch.zeros_like(adjacency))
        self.lin_in = nn.Linear(1, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, pert_mask: torch.Tensor) -> torch.Tensor:
        # pert_mask: [B, G] binary/float mask of perturbed genes
        h = self.lin_in(pert_mask.unsqueeze(-1))
        residual = h.mean(dim=1)
        if not self.use_graph:
            return residual
        adj = self.adjacency
        if self.use_gating:
            adj = adj * torch.sigmoid(self.edge_gates)
        for _ in range(self.num_layers):
            h = torch.relu(torch.einsum("ij,bjh->bih", adj, h))
            h = self.lin_msg(h)
        pooled = h.mean(dim=1)
        return pooled + residual
