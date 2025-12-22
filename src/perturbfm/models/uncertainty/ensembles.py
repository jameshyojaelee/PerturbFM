"""Deep ensemble utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

import torch


@dataclass
class DeepEnsemble:
    models: List[torch.nn.Module]

    @classmethod
    def from_factory(cls, factory: Callable[[], torch.nn.Module], k: int) -> "DeepEnsemble":
        return cls(models=[factory() for _ in range(k)])

    def predict(self, x_control: torch.Tensor, pert_idx: torch.Tensor, context_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = []
        vars_ = []
        for model in self.models:
            mean, var = model(x_control, pert_idx, context_idx)
            means.append(mean)
            vars_.append(var)
        mean_stack = torch.stack(means, dim=0)
        var_stack = torch.stack(vars_, dim=0)
        mean = mean_stack.mean(dim=0)
        aleatoric = var_stack.mean(dim=0)
        epistemic = mean_stack.var(dim=0, unbiased=False)
        total = aleatoric + epistemic
        return mean, aleatoric, total
