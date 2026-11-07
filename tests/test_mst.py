import numpy as np
import pytest

from src import benchmark, knn_graph as kg
from src.mst import ALGORITHMS, DisjointSet, boruvka, kruskal, prim
from src.mst_validation import all_passed, compare_results, scipy_reference_weight, validate_mst

# Textbook example: 6 vertices, MST = {1-2, 1-3, 3-4, 0-2, 4-5}, total weight 14
KNOWN = kg.make_graph(
    6,
    [0, 0, 1, 1, 2, 3, 4],
    [1, 2, 2, 3, 3, 4, 5],
    [4, 3, 1, 2, 4, 2, 6],
)


def random_graph(n, m, seed, max_weight=3, connected=True):
    """Many tied integer weights on purpose (stresses tie handling)."""
    rng = np.random.default_rng(seed)
    pairs = set()
    if connected:
        for i in range(1, n):
            pairs.add((int(rng.integers(0, i)), i))
    while len(pairs) < m:
        a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
        pairs.add((a, b))
    pairs = sorted(pairs)
    u, v = zip(*pairs)
    return kg.make_graph(n, u, v, rng.integers(1, max_weight + 1, size=len(pairs)).astype(float))


@pytest.mark.parametrize("name", ["Prim", "Kruskal", "Borůvka"])
def test_known_graph(name):
    result = ALGORITHMS[name](KNOWN)
    assert result.total_weight == 14
    assert result.num_edges == 5
    assert all_passed(validate_mst(KNOWN, result, scipy_reference_weight(KNOWN)))


def test_all_three_agree_and_select_identical_edges_on_known_graph():
    results = {n: f(KNOWN) for n, f in ALGORITHMS.items()}
    report = compare_results(results)
    assert report["weights_agree"] and report["identical_edge_sets"]


@pytest.mark.parametrize("seed", range(25))
def test_random_tied_graphs_match_scipy_and_each_other(seed):
    graph = random_graph(40, 120, seed)
    reference = scipy_reference_weight(graph)
    results = {n: f(graph) for n, f in ALGORITHMS.items()}
    for result in results.values():
        assert all_passed(validate_mst(graph, result, reference))
        assert result.num_edges == graph.n - 1
    report = compare_results(results)
    assert report["weights_agree"]
    assert report["identical_edge_sets"]          # (weight, id) is a strict total order


@pytest.mark.parametrize("seed", range(10))
def test_disconnected_graph_gives_minimum_spanning_forest(seed):
    graph = random_graph(30, 25, seed, connected=False)
    components, _ = kg.connected_component_labels(graph)
    assert components > 1
    reference = scipy_reference_weight(graph)
    for func in ALGORITHMS.values():
        result = func(graph)
        assert result.num_edges == graph.n - components
        assert all_passed(validate_mst(graph, result, reference))


def test_prim_start_vertex_does_not_change_result():
    graph = random_graph(40, 120, 3)
    base = frozenset(prim(graph, 0).edge_ids.tolist())
    for start in (1, 17, 39):
        assert frozenset(prim(graph, start).edge_ids.tolist()) == base


def test_boruvka_reports_phases_and_logarithmic_bound():
    graph = random_graph(64, 200, 5)
    result = boruvka(graph)
    assert result.phases == len(result.phase_ends) <= 6        # <= log2(64)
    assert result.phase_ends[-1] == result.num_edges


def test_trivial_graphs():
    empty = kg.make_graph(3, [], [], [])
    single = kg.make_graph(1, [], [], [])
    for func in ALGORITHMS.values():
        assert func(empty).num_edges == 0
        assert func(single).num_edges == 0


def test_validation_catches_bad_results():
    good = kruskal(KNOWN)
    reference = scipy_reference_weight(KNOWN)

    missing = type(good)("x", good.edge_ids[:-1], good.total_weight, 6)
    assert not all_passed(validate_mst(KNOWN, missing, reference))

    cycle_ids = np.array(sorted(set(range(7)) - {0}), dtype=np.int64)   # 6 edges -> a cycle
    cyc = type(good)("x", cycle_ids, float(KNOWN.w[cycle_ids].sum()), 6)
    assert not all_passed(validate_mst(KNOWN, cyc, reference))

    heavy = np.array([0, 1, 4, 5, 6], dtype=np.int64)    # a spanning tree, but not minimum
    tree = type(good)("x", heavy, float(KNOWN.w[heavy].sum()), 6)
    checks = validate_mst(KNOWN, tree, reference)
    failed = [name for name, ok, _ in checks if not ok]
    assert failed == ["Total weight matches the SciPy reference"]


def test_algorithms_do_not_modify_input():
    graph = random_graph(40, 120, 1)
    before = (graph.u.copy(), graph.v.copy(), graph.w.copy())
    for func in ALGORITHMS.values():
        func(graph)
        func(graph)
    assert np.array_equal(graph.u, before[0]) and np.array_equal(graph.w, before[2])


def test_disjoint_set():
    dsu = DisjointSet(5)
    assert dsu.union(0, 1) and dsu.union(1, 2) and not dsu.union(0, 2)
    assert dsu.find(0) == dsu.find(2) and dsu.components == 3


def test_real_knn_graph_all_three_agree():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(150, 16))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    graph, _ = kg.build_knn_graph(emb.astype(np.float32), 8)
    reference = scipy_reference_weight(graph)
    results = {n: f(graph) for n, f in ALGORITHMS.items()}
    assert compare_results(results)["weights_agree"]
    assert all(all_passed(validate_mst(graph, r, reference)) for r in results.values())


def test_benchmark_measures_time_and_memory():
    graph = random_graph(200, 800, 2)
    bench = benchmark.run_and_validate("Kruskal", kruskal, graph, 3, scipy_reference_weight(graph))
    assert bench.repeats == 3 and bench.median_seconds > 0
    assert bench.min_seconds <= bench.median_seconds
    assert bench.peak_memory_bytes > 0
    assert bench.valid