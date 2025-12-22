"""Probabilistic head for mean/variance output."""

from __future__ import annotations

import torch
from torch import nn


class ProbabilisticHead(nn.Module):
    def __init__(self, in_dim: int, n_genes: int):
        super().__init__()
        self.mean = nn.Linear(in_dim, n_genes)
        self.logvar = nn.Linear(in_dim, n_genes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(x)
        var = torch.nn.functional.softplus(self.logvar(x)) + 1e-6
        return mean, var
