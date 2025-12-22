"""Loss functions."""

from __future__ import annotations


def gaussian_nll(mean, var, target):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for gaussian_nll") from exc

    var = torch.clamp(var, min=1e-6)
    return 0.5 * (torch.log(var) + (target - mean) ** 2 / var).mean()
