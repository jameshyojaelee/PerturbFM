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
