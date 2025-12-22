import numpy as np
import pytest

from perturbfm.models.perturbfm.gene_graph import GeneGraph
from perturbfm.models.perturbfm.perturb_encoder import GraphPerturbationEncoder


def test_gene_graph_subset():
    genes = ["G0", "G1", "G2"]
    edges = [("G0", "G1")]
    graph = GeneGraph.from_edge_list(edges, genes)
    sub = graph.subset(["G1", "G2"])
    assert sub.adjacency.shape == (2, 2)


def test_graph_gating_zeroes_message():
    torch = pytest.importorskip("torch")
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    enc = GraphPerturbationEncoder(adjacency=adj, hidden_dim=4, use_graph=True, use_gating=True)
    enc.edge_gates.data.fill_(-20.0)
    mask = torch.tensor([[1.0, 0.0]])
    out = enc(mask)
    residual = enc.lin_in(mask.unsqueeze(-1)).mean(dim=1)
    assert torch.allclose(out, residual, atol=1e-4)
