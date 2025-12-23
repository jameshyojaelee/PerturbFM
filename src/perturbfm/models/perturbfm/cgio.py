"""Contextual Graph Intervention Operator (CGIO) model."""

from __future__ import annotations

import torch
from torch import nn


class GraphPropagator(nn.Module):
    def __init__(self, adjacencies: list[torch.Tensor], hidden_dim: int, use_gating: bool = True):
        super().__init__()
        self.adjacencies = nn.ParameterList([nn.Parameter(adj, requires_grad=False) for adj in adjacencies])
        self.use_gating = use_gating
        if use_gating:
            self.gates = nn.ParameterList([nn.Parameter(torch.zeros_like(adj)) for adj in adjacencies])
        self.lin = nn.Linear(1, hidden_dim)
        self.post = nn.Linear(hidden_dim, hidden_dim)
        self.mix = nn.Linear(hidden_dim, len(adjacencies))

    def forward(self, pert_mask: torch.Tensor, ctx_emb: torch.Tensor) -> torch.Tensor:
        # pert_mask: [B, G]; ctx_emb: [B, d]
        h0 = self.lin(pert_mask.unsqueeze(-1))  # [B, G, d]
        # Residual: pooled intervention signal before graph propagation.
        residual = h0.mean(dim=1)
        h_list = []
        for i, adj in enumerate(self.adjacencies):
            a = adj
            if self.use_gating:
                a = a * torch.sigmoid(self.gates[i])
                if torch.max(torch.abs(a)) < 1e-6:
                    h_list.append(residual)
                    continue
            h = torch.einsum("gh,bhd->bhd", a, h0)
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
        contextual_operator: bool = True,
    ):
        super().__init__()
        self.contextual_operator = contextual_operator
        self.ctx_emb = nn.Embedding(num_contexts, hidden_dim)
        self.propagator = GraphPropagator(adjacencies, hidden_dim, use_gating=use_gating)
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
