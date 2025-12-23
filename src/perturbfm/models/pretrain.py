"""Pretraining models for cell encoders."""

from __future__ import annotations

import torch
from torch import nn

from perturbfm.models.perturbfm.cell_encoder import CellEncoder


class CellAutoencoder(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = CellEncoder(n_genes, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z
