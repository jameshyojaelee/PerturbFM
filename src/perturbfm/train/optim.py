"""Optimizer utilities."""

from __future__ import annotations


def build_optimizer(params, lr: float = 1e-3, weight_decay: float = 0.0):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for build_optimizer") from exc
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
