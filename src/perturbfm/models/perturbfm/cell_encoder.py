"""Cell encoder for PerturbFM."""

from __future__ import annotations

import torch
from torch import nn


class CellEncoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerCellEncoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.gene_embed = nn.Embedding(n_genes, hidden_dim)
        self.expr_proj = nn.Linear(1, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, G]
        bsz, n_genes = x.shape
        if n_genes != self.n_genes:
            raise ValueError(f"Expected {self.n_genes} genes, got {n_genes}.")
        gene_ids = torch.arange(n_genes, device=x.device)
        gene_emb = self.gene_embed(gene_ids).unsqueeze(0).expand(bsz, -1, -1)
        expr_emb = self.expr_proj(x.unsqueeze(-1))
        tokens = gene_emb + expr_emb
        tokens = self.encoder(tokens)
        pooled = tokens.mean(dim=1)
        return self.norm(pooled)
