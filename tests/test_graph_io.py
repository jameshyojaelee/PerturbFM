import numpy as np

from perturbfm.utils.graph_io import load_graph_npz


def test_load_graph_npz_edge_index(tmp_path):
    edge_index = np.array([[0, 1], [1, 0]], dtype=np.int64)
    edge_weight = np.array([1.0, 2.0], dtype=np.float32)
    genes = np.array(["g0", "g1"], dtype=object)
    path = tmp_path / "graph.npz"
    np.savez(path, edge_index=edge_index, edge_weight=edge_weight, genes=genes)

    graph = load_graph_npz(path)
    assert isinstance(graph, dict)
    assert graph["edge_index"].shape == (2, 2)
    assert graph["edge_weight"].shape == (2,)
    assert graph["num_nodes"] == 2
