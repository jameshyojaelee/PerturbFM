import numpy as np
import pytest

from perturbfm.models.perturbfm.gene_graph import GeneGraph
from perturbfm.models.perturbfm.perturb_encoder import GraphPerturbationEncoder


def test_gene_graph_subset():
    genes = ["G0", "G1", "G2"]
    edges = [("G0", "G1")]
    graph = GeneGraph.from_edge_list(edges, genes)
    sub = graph.subset(["G1", "G2"])
    assert sub.num_nodes == 2
    assert sub.edge_index.shape[0] == 2


def test_graph_gating_zeroes_message():
    torch = pytest.importorskip("torch")
    adj = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    enc = GraphPerturbationEncoder(adjacency=adj, hidden_dim=4, use_graph=True, use_gating=True, gating_mode="scalar")
    enc.gate_scalar.data.fill_(-20.0)
    mask = torch.tensor([[1.0, 0.0]])
    out = enc(mask)
    residual = enc.lin_in(mask.unsqueeze(-1)).mean(dim=1)
    assert torch.allclose(out, residual, atol=1e-4)


def test_no_dense_gate_matrices():
    torch = pytest.importorskip("torch")
    g = 16
    adj = torch.eye(g)
    enc = GraphPerturbationEncoder(adjacency=adj, hidden_dim=4, gating_mode="lowrank", gate_rank=4)
    for p in enc.parameters():
        if p.ndim == 2 and p.shape[0] == p.shape[1] == g:
            raise AssertionError("Dense GxG gate parameter found in GraphPerturbationEncoder")

    from perturbfm.models.perturbfm.cgio import GraphPropagator

    prop = GraphPropagator([adj], hidden_dim=4, gating_mode="lowrank", gate_rank=4)
    for p in prop.parameters():
        if p.ndim == 2 and p.shape[0] == p.shape[1] == g:
            raise AssertionError("Dense GxG gate parameter found in GraphPropagator")


def test_graph_propagator_accepts_sparse():
    torch = pytest.importorskip("torch")
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float32)
    sparse_adj = torch.sparse_coo_tensor(edge_index, edge_weight, (2, 2))
    from perturbfm.models.perturbfm.cgio import GraphPropagator

    prop = GraphPropagator([sparse_adj], hidden_dim=4, gating_mode="scalar")
    pert_mask = torch.zeros((1, 2), dtype=torch.float32)
    ctx_emb = torch.zeros((1, 4), dtype=torch.float32)
    out = prop(pert_mask, ctx_emb)
    assert out.shape == (1, 4)
