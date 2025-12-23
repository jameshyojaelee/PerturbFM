import pytest


def test_cgio_graphpropagator_zero_gate_uses_residual():
    torch = pytest.importorskip("torch")
    from perturbfm.models.perturbfm.cgio import GraphPropagator

    g = 4
    adj = torch.ones((g, g))
    prop = GraphPropagator([adj], hidden_dim=8, use_gating=True)
    prop.gates[0].data.fill_(-20.0)
    pert_mask = torch.zeros((2, g))
    pert_mask[0, 0] = 1.0
    ctx = torch.zeros((2, 8))

    out = prop(pert_mask, ctx)
    residual = prop.lin(pert_mask.unsqueeze(-1)).mean(dim=1)
    assert torch.allclose(out, residual, atol=1e-4)
