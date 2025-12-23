"""Contextual Graph Intervention Operator (CGIO) model."""

from __future__ import annotations

import torch
from torch import nn


class GraphPropagator(nn.Module):
    def __init__(
        self,
        adjacencies: list[torch.Tensor],
        hidden_dim: int,
        use_gating: bool = True,
        gating_mode: str | None = None,
        gate_rank: int = 16,
        gate_hidden: int = 32,
    ):
        super().__init__()
        self.lin = nn.Linear(1, hidden_dim)
        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.mix = nn.Linear(hidden_dim, len(adjacencies))

        # Convert dense adjacencies to edge_index/edge_weight
        self.edges = []
        self.num_nodes = []
        for adj in adjacencies:
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            edge_weight = adj[edge_index[0], edge_index[1]]
            self.edges.append((edge_index, edge_weight))
            self.num_nodes.append(adj.shape[0])

        if gating_mode is None:
            gating_mode = "scalar" if use_gating else "none"
        self.gating_mode = gating_mode
        self._init_gates(gate_rank, gate_hidden)

    def _init_gates(self, gate_rank: int, gate_hidden: int) -> None:
        mode = self.gating_mode
        if mode == "none":
            self.gate_scalars = None
        elif mode == "scalar":
            self.gate_scalars = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in self.edges])
        elif mode == "node":
            self.gate_nodes = nn.ParameterList([nn.Parameter(torch.zeros(n)) for n in self.num_nodes])
        elif mode == "lowrank":
            self.gate_u = nn.ParameterList([nn.Parameter(torch.zeros(n, gate_rank)) for n in self.num_nodes])
            self.gate_v = nn.ParameterList([nn.Parameter(torch.zeros(n, gate_rank)) for n in self.num_nodes])
        elif mode == "mlp":
            self.gate_nodes = nn.ParameterList([nn.Parameter(torch.zeros(n, gate_hidden)) for n in self.num_nodes])
            self.gate_mlps = nn.ModuleList(
                [nn.Sequential(nn.Linear(gate_hidden * 2, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1)) for _ in self.num_nodes]
            )
        else:
            raise ValueError(f"Unknown gating_mode: {mode}")

    def _edge_gates(self, i: int, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        mode = self.gating_mode
        if mode == "none":
            return torch.ones_like(edge_weight)
        if mode == "scalar":
            return torch.sigmoid(self.gate_scalars[i]) * torch.ones_like(edge_weight)
        src, dst = edge_index
        if mode == "node":
            g = torch.sigmoid(self.gate_nodes[i])
            return g[src] * g[dst]
        if mode == "lowrank":
            u = self.gate_u[i][src]
            v = self.gate_v[i][dst]
            return torch.sigmoid((u * v).sum(dim=-1))
        if mode == "mlp":
            emb = self.gate_nodes[i]
            x = torch.cat([emb[src], emb[dst]], dim=-1)
            return torch.sigmoid(self.gate_mlps[i](x).squeeze(-1))
        return torch.ones_like(edge_weight)

    def _spmm(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, h: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        w = edge_weight * gate
        out = torch.zeros_like(h)
        for b in range(h.shape[0]):
            msg = h[b, src] * w.unsqueeze(-1)
            out[b].index_add_(0, dst, msg)
        return out

    def forward(self, pert_mask: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        # pert_mask: [B, G]; ctx_emb: [B, d]
        h0 = self.lin(pert_mask.unsqueeze(-1))  # [B, G, d]
        residual = h0.mean(dim=1)
        h_list = []
        for i, (edge_index, edge_weight) in enumerate(self.edges):
            if edge_index.numel() == 0:
                h_list.append(residual)
                continue
            gate = self._edge_gates(i, edge_index, edge_weight)
            if gate.numel() == 0 or torch.max(torch.abs(gate)) < 1e-6:
                h_list.append(residual)
                continue
            h = self._spmm(edge_index, edge_weight, h0, gate)
            h = torch.relu(self.post(h))
            h = h.mean(dim=1)  # pool genes -> [B, d]
            h_list.append(h)
        H = torch.stack(h_list, dim=1)  # [B, M, d]
        weights = torch.softmax(self.mix(ctx_emb), dim=1).unsqueeze(-1)  # [B, M, 1]
        return (H * weights).sum(dim=1)  # [B, d]


class CGIO(nn.Module):
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int,
        num_contexts: int,
        adjacencies: list[torch.Tensor],
        num_bases: int = 4,
        use_gating: bool = True,
        gating_mode: str | None = None,
        contextual_operator: bool = True,
    ):
        super().__init__()
        self.contextual_operator = contextual_operator
        self.ctx_emb = nn.Embedding(num_contexts, hidden_dim)
        self.propagator = GraphPropagator(
            adjacencies,
            hidden_dim,
            use_gating=use_gating,
            gating_mode=gating_mode,
        )
        self.bases = nn.Parameter(torch.randn(num_bases, n_genes, hidden_dim) * 0.02)
        self.alpha = nn.Linear(hidden_dim, num_bases)
        self.var_head = nn.Linear(hidden_dim * 2, n_genes)

    def forward(self, pert_mask: torch.Tensor, context_idx: torch.Tensor):
        ctx = self.ctx_emb(context_idx)
        h = self.propagator(pert_mask, ctx)
        if self.contextual_operator:
            alphas = torch.softmax(self.alpha(ctx), dim=-1)  # [B, K]
        else:
            alphas = torch.softmax(self.alpha.weight.mean(dim=0), dim=-1).unsqueeze(0).expand(h.size(0), -1)
        # mean = sum_k alpha_k * (h @ B_k^T)
        means = []
        for k in range(self.bases.size(0)):
            Bk = self.bases[k]  # [G, d]
            means.append(torch.matmul(h, Bk.t()))  # [B, G]
        mean_stack = torch.stack(means, dim=1)  # [B, K, G]
        mean = (mean_stack * alphas.unsqueeze(-1)).sum(dim=1)
        var = torch.nn.functional.softplus(self.var_head(torch.cat([h, ctx], dim=-1))) + 1e-6
        return mean, var
