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
        adjacency: torch.Tensor | None = None,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_graph: bool = True,
        use_gating: bool = True,
        edge_index: torch.Tensor | None = None,
        edge_weight: torch.Tensor | None = None,
        num_nodes: int | None = None,
        gating_mode: str | None = None,
        gate_rank: int = 16,
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.use_graph = use_graph
        self.num_layers = num_layers

        if edge_index is None:
            if adjacency is None:
                raise ValueError("Provide adjacency or edge_index.")
            edge_index = adjacency.nonzero(as_tuple=False).t().contiguous()
            edge_weight = adjacency[edge_index[0], edge_index[1]]
            num_nodes = adjacency.shape[0]
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
        if num_nodes is None:
            num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.num_nodes = num_nodes

        if gating_mode is None:
            gating_mode = "scalar" if use_gating else "none"
        self.gating_mode = gating_mode
        self._init_gates(gate_rank, gate_hidden)

        self.lin_in = nn.Linear(1, hidden_dim)
        self.lin_msg = nn.Linear(hidden_dim, hidden_dim)

    def _init_gates(self, gate_rank: int, gate_hidden: int) -> None:
        mode = self.gating_mode
        if mode == "none":
            self.gate_scalar = None
        elif mode == "scalar":
            self.gate_scalar = nn.Parameter(torch.zeros(1))
        elif mode == "node":
            self.gate_node = nn.Parameter(torch.zeros(self.num_nodes))
        elif mode == "lowrank":
            self.gate_u = nn.Parameter(torch.zeros(self.num_nodes, gate_rank))
            self.gate_v = nn.Parameter(torch.zeros(self.num_nodes, gate_rank))
        elif mode == "mlp":
            self.gate_node = nn.Parameter(torch.zeros(self.num_nodes, gate_hidden))
            self.gate_mlp = nn.Sequential(
                nn.Linear(gate_hidden * 2, gate_hidden),
                nn.ReLU(),
                nn.Linear(gate_hidden, 1),
            )
        else:
            raise ValueError(f"Unknown gating_mode: {mode}")

    def _edge_gates(self) -> torch.Tensor:
        mode = self.gating_mode
        if mode == "none":
            return torch.ones(self.edge_weight.shape, device=self.edge_weight.device)
        if mode == "scalar":
            return torch.sigmoid(self.gate_scalar) * torch.ones_like(self.edge_weight)
        src, dst = self.edge_index
        if mode == "node":
            g = torch.sigmoid(self.gate_node)
            return g[src] * g[dst]
        if mode == "lowrank":
            u = self.gate_u[src]
            v = self.gate_v[dst]
            return torch.sigmoid((u * v).sum(dim=-1))
        if mode == "mlp":
            emb = self.gate_node
            x = torch.cat([emb[src], emb[dst]], dim=-1)
            return torch.sigmoid(self.gate_mlp(x).squeeze(-1))
        return torch.ones_like(self.edge_weight)

    def _spmm(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, G, d]
        src, dst = self.edge_index
        w = self.edge_weight * self._edge_gates()
        if w.numel() == 0 or torch.max(torch.abs(w)) < 1e-6:
            return torch.zeros_like(h)
        out = torch.zeros_like(h)
        for b in range(h.shape[0]):
            msg = h[b, src] * w.unsqueeze(-1)
            out[b].index_add_(0, dst, msg)
        return out

    def forward(self, pert_mask: torch.Tensor) -> torch.Tensor:
        # pert_mask: [B, G] binary/float mask of perturbed genes
        h = self.lin_in(pert_mask.unsqueeze(-1))
        residual = h.mean(dim=1)
        if not self.use_graph:
            return residual
        # If gating fully suppresses edges, return residual
        if self.edge_index.numel() == 0:
            return residual
        w = self.edge_weight * self._edge_gates()
        if w.numel() == 0 or torch.max(torch.abs(w)) < 1e-6:
            return residual
        for _ in range(self.num_layers):
            h = torch.relu(self._spmm(h))
            h = self.lin_msg(h)
        pooled = h.mean(dim=1)
        return pooled + residual
