"""Dataset registry and synthetic dataset utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.utils.seeds import set_seed


LoaderFn = Callable[..., PerturbDataset]


@dataclass
class DatasetRegistry:
    loaders: Dict[str, LoaderFn] = field(default_factory=dict)

    def register(self, name: str, loader_fn: LoaderFn) -> None:
        if name in self.loaders:
            raise ValueError(f"Dataset loader already registered: {name}")
        self.loaders[name] = loader_fn

    def load(self, name: str, **kwargs: Any) -> PerturbDataset:
        if name not in self.loaders:
            raise KeyError(f"Unknown dataset loader: {name}")
        return self.loaders[name](**kwargs)


def make_synthetic_dataset(
    n_obs: int = 200,
    n_genes: int = 50,
    n_contexts: int = 3,
    n_perts: int = 5,
    seed: int = 0,
) -> PerturbDataset:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    contexts = [f"C{i}" for i in range(n_contexts)]
    perts = [f"P{i}" for i in range(n_perts)]

    base_context = rng.normal(0.0, 1.0, size=(n_contexts, n_genes))
    pert_effect = rng.normal(0.0, 0.5, size=(n_perts, n_genes))

    obs = {k: [] for k in ("pert_id", "context_id", "batch_id", "is_control", "pert_genes")}
    X_control = np.zeros((n_obs, n_genes), dtype=np.float32)
    X_pert = np.zeros((n_obs, n_genes), dtype=np.float32)

    # Random sparse undirected adjacency for tests (normalized)
    adj = rng.random((n_genes, n_genes))
    adj = np.triu(adj < 0.05, 1).astype(np.float32)
    adj = adj + adj.T
    deg = adj.sum(axis=1, keepdims=True) + 1e-6
    adj = adj / deg

    for i in range(n_obs):
        ctx_idx = rng.integers(0, n_contexts)
        is_control = rng.random() < 0.2
        pert_idx = 0 if is_control else rng.integers(0, n_perts)

        control = base_context[ctx_idx] + rng.normal(0.0, 0.1, size=n_genes)
        effect = np.zeros(n_genes) if is_control else pert_effect[pert_idx]
        perturbed = control + effect + rng.normal(0.0, 0.05, size=n_genes)

        X_control[i] = control
        X_pert[i] = perturbed

        obs["pert_id"].append("control" if is_control else perts[pert_idx])
        obs["context_id"].append(contexts[ctx_idx])
        obs["batch_id"].append("B0")
        obs["is_control"].append(bool(is_control))
        if is_control:
            obs["pert_genes"].append([])
        else:
            # pick 1-3 genes for the perturbation
            gcount = int(rng.integers(1, min(3, n_genes) + 1))
            genes = rng.choice(n_genes, size=gcount, replace=False)
            obs["pert_genes"].append([f"G{g}" for g in genes])

    delta = X_pert - X_control
    var = [f"G{i}" for i in range(n_genes)]
    metadata = {"name": "synthetic", "n_contexts": n_contexts, "n_perts": n_perts, "adjacency": adj.tolist()}
    ds = PerturbDataset(X_control=X_control, X_pert=X_pert, delta=delta, obs=obs, var=var, metadata=metadata)
    ds.validate()
    return ds


def load_artifact(path: str) -> PerturbDataset:
    return PerturbDataset.load_artifact(path)


registry = DatasetRegistry()
registry.register("synthetic", make_synthetic_dataset)
