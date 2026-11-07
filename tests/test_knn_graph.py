import numpy as np
import pytest

from src import config
from src import knn_graph as kg
from src.embeddings import l2_normalize


def random_unit(n=60, d=12, seed=0):
    return l2_normalize(np.random.default_rng(seed).normal(size=(n, d)))


def test_knn_matches_brute_force():
    emb = random_unit()
    k = 5
    got = kg.knn_search(emb, k, chunk_size=17)       # odd chunk size on purpose
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    expected = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    assert np.array_equal(got, expected)


def test_knn_matches_scikit_learn():
    neighbors = pytest.importorskip("sklearn.neighbors")
    emb = random_unit()
    k = 6
    model = neighbors.NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute").fit(emb)
    _, idx = model.kneighbors(emb)
    assert np.array_equal(kg.knn_search(emb, k), idx[:, 1:])


def test_no_self_neighbours_and_bad_k_rejected():
    emb = random_unit()
    out = kg.knn_search(emb, 4)
    assert all(i not in row for i, row in enumerate(out))
    with pytest.raises(ValueError):
        kg.knn_search(emb, len(emb))


def test_union_symmetrization_on_handmade_table():
    # 0 chose 1; 1 chose 0 (mutual); 2 chose 1 (one-sided); 3 chose 0 (one-sided)
    neighbours = np.array([[1], [0], [1], [0]])
    u, v, info = kg.symmetrize_union(4, neighbours)
    assert list(zip(u.tolist(), v.tolist())) == [(0, 1), (0, 3), (1, 2)]
    assert info == {"directed_selections": 4, "self_loops_removed": 0, "mutual_pairs_merged": 1}


def test_self_loops_are_removed():
    neighbours = np.array([[0], [0]])        # vertex 0 lists itself
    u, v, info = kg.symmetrize_union(2, neighbours)
    assert list(zip(u.tolist(), v.tolist())) == [(0, 1)]
    assert info["self_loops_removed"] == 1


def test_weights_are_cosine_distance_and_u_less_than_v():
    emb = random_unit()
    graph, stats = kg.build_knn_graph(emb, 5)
    assert (graph.u < graph.v).all()
    assert len(set(zip(graph.u.tolist(), graph.v.tolist()))) == graph.m      # no duplicates
    manual = 1.0 - np.einsum("ij,ij->i", emb[graph.u].astype(float), emb[graph.v].astype(float))
    assert np.allclose(graph.w, np.clip(manual, 0, 2))
    assert stats["edges"] == graph.m
    assert stats["directed_selections"] - stats["mutual_pairs_merged"] - stats["self_loops_removed"] == graph.m


def test_statistics_and_disconnected_detection():
    # two groups pointing in orthogonal directions: with k=2 no edge can cross
    a = l2_normalize(np.array([[1, 0.01 * i, 0] for i in range(5)]))
    b = l2_normalize(np.array([[0, 1, 0.01 * i] for i in range(5)]))
    graph, stats = kg.build_knn_graph(np.vstack([a, b]), 2)
    assert stats["components"] == 2 and stats["connected"] is False
    assert sorted(stats["component_sizes"]) == [5, 5]
    assert stats["density"] == pytest.approx(2 * graph.m / (10 * 9))


def test_connected_graph_reported_connected():
    graph, stats = kg.build_knn_graph(random_unit(), 10)
    assert stats["connected"] is True and stats["components"] == 1


def test_graph_arrays_are_read_only():
    graph, _ = kg.build_knn_graph(random_unit(), 5)
    with pytest.raises(ValueError):
        graph.w[0] = 99.0


def test_save_and_load_round_trip_and_stale_key(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GRAPHS_DIR", tmp_path)
    graph, stats = kg.build_knn_graph(random_unit(), 5)
    kg.save_graph(graph, stats, "key-A")
    loaded, loaded_stats = kg.load_graph(5, "key-A")
    assert np.array_equal(loaded.u, graph.u) and np.allclose(loaded.w, graph.w)
    assert loaded_stats["edges"] == stats["edges"]
    assert kg.load_graph(5, "key-B") is None
    assert kg.saved_graph_ks("key-A") == [5] and kg.saved_graph_ks("key-B") == []
    assert kg.load_graph(10, "key-A") is None


def test_neighbourhood_and_induced_subgraph():
    # path 0-1-2-3-4
    graph = kg.make_graph(5, [0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4])
    adjacency = kg.neighbour_lists(graph)
    nodes, depth = kg.bfs_neighbourhood(adjacency, 2, hops=1, max_nodes=10)
    assert sorted(nodes) == [1, 2, 3] and depth[2] == 0
    nodes, _ = kg.bfs_neighbourhood(adjacency, 2, hops=5, max_nodes=3)
    assert len(nodes) == 3
    sub = kg.induced_subgraph(graph, [1, 2, 3])
    assert sub.graph.n == 3 and sub.graph.m == 2
    assert sub.original_edge_ids.tolist() == [1, 2]