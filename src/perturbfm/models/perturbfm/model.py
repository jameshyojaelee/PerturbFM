"""PerturbFM v0 model (probabilistic, no graph)."""

from __future__ import annotations

import torch
from torch import nn

from perturbfm.models.perturbfm.cell_encoder import CellEncoder, TransformerCellEncoder
from perturbfm.models.perturbfm.fusion import MLP
from perturbfm.models.perturbfm.perturb_encoder import PerturbationEncoder
from perturbfm.models.perturbfm.cgio import GraphPropagator


class PerturbFMv0(nn.Module):
    def __init__(
        self,
        n_genes: int,
        num_perts: int,
        num_contexts: int,
        hidden_dim: int = 128,
        use_basal: bool = True,
        use_context: bool = True,
        use_perturbation: bool = True,
    ):
        super().__init__()
        self.use_basal = use_basal
        self.use_context = use_context
        self.use_perturbation = use_perturbation

        self.cell_encoder = CellEncoder(n_genes, hidden_dim)
        self.pert_encoder = PerturbationEncoder(num_perts, hidden_dim)
        self.context_encoder = nn.Embedding(num_contexts, hidden_dim)

        self.basal_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.context_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.pert_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.var_head = MLP(hidden_dim * 3, n_genes, hidden_dim)

    def forward(
        self,
        x_control: torch.Tensor,
        pert_idx: torch.Tensor,
        context_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cell_emb = self.cell_encoder(x_control)
        pert_emb = self.pert_encoder(pert_idx)
        ctx_emb = self.context_encoder(context_idx)

        mean = torch.zeros_like(x_control)
        if self.use_basal:
            mean = mean + self.basal_head(cell_emb)
        if self.use_context:
            mean = mean + self.context_head(ctx_emb)
        if self.use_perturbation:
            mean = mean + self.pert_head(pert_emb)

        fused = torch.cat([cell_emb, pert_emb, ctx_emb], dim=-1)
        var = torch.nn.functional.softplus(self.var_head(fused)) + 1e-6
        return mean, var


class PerturbFMv1(nn.Module):
    def __init__(
        self,
        n_genes: int,
        num_contexts: int,
        adjacency: torch.Tensor,
        hidden_dim: int = 128,
        use_graph: bool = True,
        use_gating: bool = True,
        gating_mode: str | None = None,
    ):
        super().__init__()
        from perturbfm.models.perturbfm.perturb_encoder import GraphPerturbationEncoder

        self.cell_encoder = CellEncoder(n_genes, hidden_dim)
        self.context_encoder = nn.Embedding(num_contexts, hidden_dim)
        self.pert_encoder = GraphPerturbationEncoder(
            adjacency=adjacency,
            hidden_dim=hidden_dim,
            use_graph=use_graph,
            use_gating=use_gating,
            gating_mode=gating_mode,
        )

        self.basal_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.context_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.pert_head = MLP(hidden_dim, n_genes, hidden_dim)
        self.var_head = MLP(hidden_dim * 3, n_genes, hidden_dim)

    def forward(self, x_control: torch.Tensor, pert_mask: torch.Tensor, context_idx: torch.Tensor):
        cell_emb = self.cell_encoder(x_control)
        ctx_emb = self.context_encoder(context_idx)
        pert_emb = self.pert_encoder(pert_mask)

        mean = self.basal_head(cell_emb) + self.context_head(ctx_emb) + self.pert_head(pert_emb)
        fused = torch.cat([cell_emb, pert_emb, ctx_emb], dim=-1)
        var = torch.nn.functional.softplus(self.var_head(fused)) + 1e-6
        return mean, var


class PerturbFMv3(nn.Module):
    def __init__(
        self,
        n_genes: int,
        num_contexts: int,
        hidden_dim: int,
        adjacencies: list[torch.Tensor],
        use_gating: bool = True,
        gating_mode: str | None = None,
    ):
        super().__init__()
        self.cell_encoder = CellEncoder(n_genes, hidden_dim)
        self.context_encoder = nn.Embedding(num_contexts, hidden_dim)
        self.pert_encoder = GraphPropagator(
            adjacencies=adjacencies,
            hidden_dim=hidden_dim,
            use_gating=use_gating,
            gating_mode=gating_mode,
        )
        self.fuse = MLP(hidden_dim * 3, n_genes, hidden_dim)
        self.var_head = MLP(hidden_dim * 3, n_genes, hidden_dim)

    def forward(self, x_control: torch.Tensor, pert_mask: torch.Tensor, context_idx: torch.Tensor):
        basal = self.cell_encoder(x_control)
        ctx_emb = self.context_encoder(context_idx)
        pert_emb = self.pert_encoder(pert_mask, ctx_emb)
        fused = torch.cat([basal, pert_emb, ctx_emb], dim=-1)
        mean = self.fuse(fused)
        var = torch.nn.functional.softplus(self.var_head(fused)) + 1e-6
        return mean, var


class PerturbFMv3a(nn.Module):
    def __init__(
        self,
        n_genes: int,
        num_contexts: int,
        hidden_dim: int,
        adjacencies: list[torch.Tensor],
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_gating: bool = True,
        gating_mode: str | None = None,
    ):
        super().__init__()
        self.cell_encoder = TransformerCellEncoder(
            n_genes=n_genes,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.context_encoder = nn.Embedding(num_contexts, hidden_dim)
        self.pert_encoder = GraphPropagator(
            adjacencies=adjacencies,
            hidden_dim=hidden_dim,
            use_gating=use_gating,
            gating_mode=gating_mode,
        )
        self.fuse = MLP(hidden_dim * 3, n_genes, hidden_dim)
        self.var_head = MLP(hidden_dim * 3, n_genes, hidden_dim)

    def forward(self, x_control: torch.Tensor, pert_mask: torch.Tensor, context_idx: torch.Tensor):
        basal = self.cell_encoder(x_control)
        ctx_emb = self.context_encoder(context_idx)
        pert_emb = self.pert_encoder(pert_mask, ctx_emb)
        fused = torch.cat([basal, pert_emb, ctx_emb], dim=-1)
        mean = self.fuse(fused)
        var = torch.nn.functional.softplus(self.var_head(fused)) + 1e-6
        return mean, var
