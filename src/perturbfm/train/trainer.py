"""Training utilities."""

from __future__ import annotations

from typing import Dict, Tuple, List

import numpy as np

from perturbfm.data.canonical import PerturbDataset
from perturbfm.data.splits.split_spec import Split
from perturbfm.models.baselines.mean_delta import MeanDeltaBaseline
from perturbfm.models.baselines.ridge_delta import RidgeDeltaBaseline
from perturbfm.models.perturbfm.model import PerturbFMv0, PerturbFMv1
from perturbfm.models.perturbfm.cgio import CGIO
from perturbfm.train.losses import gaussian_nll
from perturbfm.train.optim import build_optimizer
from perturbfm.data.transforms import pert_genes_to_mask


def get_baseline(name: str, **kwargs):
    if name == "global_mean":
        return MeanDeltaBaseline(mode="global")
    if name == "per_perturbation_mean":
        return MeanDeltaBaseline(mode="per_perturbation")
    if name == "per_perturbation_context_mean":
        return MeanDeltaBaseline(mode="per_perturbation_context")
    if name == "ridge":
        return RidgeDeltaBaseline(alpha=float(kwargs.get("alpha", 1.0)))
    raise ValueError(f"Unknown baseline: {name}")


def _fit_baseline(model, dataset: PerturbDataset, split: Split) -> None:
    if isinstance(model, MeanDeltaBaseline):
        model.fit(dataset.delta, dataset.obs, split.train_idx)
        return
    if isinstance(model, RidgeDeltaBaseline):
        if dataset.X_control is None:
            raise ValueError("Ridge baseline requires X_control.")
        model.fit(dataset.X_control, dataset.delta, split.train_idx)
        return
    raise ValueError("Unsupported baseline model type.")


def _predict_baseline(model, dataset: PerturbDataset, idx: np.ndarray) -> np.ndarray:
    if isinstance(model, MeanDeltaBaseline):
        return model.predict(dataset.obs, idx)
    if isinstance(model, RidgeDeltaBaseline):
        return model.predict(dataset.X_control, idx)
    raise ValueError("Unsupported baseline model type.")


def fit_predict_baseline(model, dataset: PerturbDataset, split: Split) -> Dict[str, np.ndarray]:
    _fit_baseline(model, dataset, split)
    train_pred = _predict_baseline(model, dataset, split.train_idx)
    residual = train_pred - dataset.delta[split.train_idx]
    var = residual.var(axis=0, ddof=1) + 1e-6
    test_pred = _predict_baseline(model, dataset, split.test_idx)
    var_broadcast = np.tile(var, (len(split.test_idx), 1))
    return {"mean": test_pred, "var": var_broadcast, "idx": split.test_idx, "model": model}


def _build_index(values):
    uniq = sorted(set(values))
    return {v: i for i, v in enumerate(uniq)}


def fit_predict_perturbfm_v0(
    dataset: PerturbDataset,
    split: Split,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cpu",
    use_basal: bool = True,
    use_context: bool = True,
    use_perturbation: bool = True,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for PerturbFMv0 training") from exc

    pert_map = _build_index(dataset.obs["pert_id"])
    ctx_map = _build_index(dataset.obs["context_id"])
    pert_idx_all = np.array([pert_map[p] for p in dataset.obs["pert_id"]], dtype=np.int64)
    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)

    model = PerturbFMv0(
        n_genes=dataset.n_genes,
        num_perts=len(pert_map),
        num_contexts=len(ctx_map),
        hidden_dim=hidden_dim,
        use_basal=use_basal,
        use_context=use_context,
        use_perturbation=use_perturbation,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        idx = split.train_idx
        x = torch.as_tensor(dataset.X_control[idx], dtype=torch.float32, device=device)
        y = torch.as_tensor(dataset.delta[idx], dtype=torch.float32, device=device)
        p = torch.as_tensor(pert_idx_all[idx], dtype=torch.long, device=device)
        c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long, device=device)
        mean, var = model(x, p, c)
        loss = gaussian_nll(mean, var, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_idx = split.test_idx
        x = torch.as_tensor(dataset.X_control[test_idx], dtype=torch.float32, device=device)
        p = torch.as_tensor(pert_idx_all[test_idx], dtype=torch.long, device=device)
        c = torch.as_tensor(ctx_idx_all[test_idx], dtype=torch.long, device=device)
        mean, var = model(x, p, c)
    return {
        "model": model,
        "mean": mean.cpu().numpy(),
        "var": var.cpu().numpy(),
        "idx": test_idx,
        "pert_map": pert_map,
        "ctx_map": ctx_map,
    }


def fit_predict_perturbfm_v1(
    dataset: PerturbDataset,
    split: Split,
    adjacency,
    pert_gene_masks: Dict[str, np.ndarray],
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cpu",
    use_graph: bool = True,
    use_gating: bool = True,
    gating_mode: str | None = None,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for PerturbFMv1 training") from exc

    ctx_map = _build_index(dataset.obs["context_id"])
    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)

    pert_masks = []
    for pert in dataset.obs["pert_id"]:
        mask = pert_gene_masks.get(pert)
        if mask is None:
            mask = np.zeros(dataset.n_genes, dtype=np.float32)
        pert_masks.append(mask)
    pert_masks = np.stack(pert_masks, axis=0).astype(np.float32)

    model = PerturbFMv1(
        n_genes=dataset.n_genes,
        num_contexts=len(ctx_map),
        adjacency=torch.as_tensor(adjacency, dtype=torch.float32),
        hidden_dim=hidden_dim,
        use_graph=use_graph,
        use_gating=use_gating,
        gating_mode=gating_mode,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        idx = split.train_idx
        x = torch.as_tensor(dataset.X_control[idx], dtype=torch.float32, device=device)
        y = torch.as_tensor(dataset.delta[idx], dtype=torch.float32, device=device)
        p = torch.as_tensor(pert_masks[idx], dtype=torch.float32, device=device)
        c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long, device=device)
        mean, var = model(x, p, c)
        loss = gaussian_nll(mean, var, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_idx = split.test_idx
        x = torch.as_tensor(dataset.X_control[test_idx], dtype=torch.float32, device=device)
        p = torch.as_tensor(pert_masks[test_idx], dtype=torch.float32, device=device)
        c = torch.as_tensor(ctx_idx_all[test_idx], dtype=torch.long, device=device)
        mean, var = model(x, p, c)

    return {
        "model": model,
        "mean": mean.cpu().numpy(),
        "var": var.cpu().numpy(),
        "idx": test_idx,
        "ctx_map": ctx_map,
    }


def fit_predict_perturbfm_v2(
    dataset: PerturbDataset,
    split: Split,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    epochs: int = 50,
    device: str = "cpu",
    use_gating: bool = True,
    gating_mode: str | None = None,
    contextual_operator: bool = True,
    num_bases: int = 4,
    adjacencies: List[np.ndarray] | None = None,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for PerturbFMv2 training") from exc

    ctx_map = _build_index(dataset.obs["context_id"])
    ctx_idx_all = np.array([ctx_map[c] for c in dataset.obs["context_id"]], dtype=np.int64)
    pert_mask = pert_genes_to_mask(dataset.obs.get("pert_genes", [[] for _ in range(dataset.n_obs)]), dataset.var)

    if adjacencies is None:
        adj_meta = dataset.metadata.get("adjacency")
        if adj_meta is None:
            raise ValueError("Adjacency required for CGIO (provide via metadata or parameter).")
        adjacencies = [np.asarray(adj_meta, dtype=np.float32)]
    adj_tensors = [torch.as_tensor(a, dtype=torch.float32, device=device) for a in adjacencies]

    model = CGIO(
        n_genes=dataset.n_genes,
        hidden_dim=hidden_dim,
        num_contexts=len(ctx_map),
        adjacencies=adj_tensors,
        num_bases=num_bases,
        use_gating=use_gating,
        gating_mode=gating_mode,
        contextual_operator=contextual_operator,
    ).to(device)

    optimizer = build_optimizer(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        idx = split.train_idx
        pm = torch.as_tensor(pert_mask[idx], dtype=torch.float32, device=device)
        y = torch.as_tensor(dataset.delta[idx], dtype=torch.float32, device=device)
        c = torch.as_tensor(ctx_idx_all[idx], dtype=torch.long, device=device)
        mean, var = model(pm, c)
        loss = gaussian_nll(mean, var, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_idx = split.test_idx
        pm = torch.as_tensor(pert_mask[test_idx], dtype=torch.float32, device=device)
        c = torch.as_tensor(ctx_idx_all[test_idx], dtype=torch.long, device=device)
        mean, var = model(pm, c)

    return {
        "model": model,
        "mean": mean.cpu().numpy(),
        "var": var.cpu().numpy(),
        "idx": test_idx,
        "ctx_map": ctx_map,
    }
