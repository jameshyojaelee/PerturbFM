"""Additive mean baseline (sum of single-gene mean deltas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class AdditiveMeanBaseline:
    gene_means: Dict[str, np.ndarray] = field(default_factory=dict)
    missing_genes: List[str] = field(default_factory=list)
    missing_contribs: int = 0

    def _normalize_genes(self, genes) -> List[str]:
        if genes is None:
            return []
        if isinstance(genes, str):
            return [genes]
        if isinstance(genes, np.ndarray):
            genes = genes.tolist()
        return [str(g) for g in genes]

    def fit(self, delta: np.ndarray, obs: dict, idx: np.ndarray) -> None:
        if delta is None:
            raise ValueError("additive_mean baseline requires delta.")
        if "pert_genes" not in obs:
            raise ValueError("additive_mean baseline requires obs['pert_genes'].")
        idx = np.asarray(idx, dtype=np.int64)

        per_gene_sum: Dict[str, np.ndarray] = {}
        per_gene_count: Dict[str, int] = {}
        pert_genes = obs["pert_genes"]
        for i in idx:
            genes = self._normalize_genes(pert_genes[i])
            if len(genes) != 1:
                continue
            gene = genes[0]
            if gene not in per_gene_sum:
                per_gene_sum[gene] = np.array(delta[i], copy=True)
                per_gene_count[gene] = 1
            else:
                per_gene_sum[gene] += delta[i]
                per_gene_count[gene] += 1

        if not per_gene_sum:
            raise ValueError("additive_mean baseline found no single-gene perturbations in training indices.")

        self.gene_means = {g: per_gene_sum[g] / per_gene_count[g] for g in per_gene_sum}

    def predict(self, obs: dict, idx: np.ndarray) -> np.ndarray:
        if not self.gene_means:
            raise RuntimeError("Model is not fit.")
        if "pert_genes" not in obs:
            raise ValueError("additive_mean baseline requires obs['pert_genes'].")
        idx = np.asarray(idx, dtype=np.int64)

        example = next(iter(self.gene_means.values()))
        out = np.zeros((len(idx), example.shape[0]), dtype=example.dtype)
        missing: set[str] = set()
        missing_contribs = 0

        pert_genes = obs["pert_genes"]
        for row, i in enumerate(idx):
            genes = self._normalize_genes(pert_genes[i])
            if not genes:
                continue
            for gene in genes:
                mean = self.gene_means.get(gene)
                if mean is None:
                    missing.add(gene)
                    missing_contribs += 1
                    continue
                out[row] += mean

        self.missing_genes = sorted(missing)
        self.missing_contribs = missing_contribs
        return out
