"""Tests for Experiment C (Complete Graph vs kNN Graph)."""
import numpy as np
import pytest

from src.dense_vs_knn import build_complete_graph, run_experiment_c
from src.knn_graph import make_graph
from src.mst import prim


def test_build_complete_graph():
    # 10 random 4D unit embeddings
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((10, 4)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    graph, elapsed = build_complete_graph(mat)
    assert graph.n == 10
    expected_edges = 10 * 9 // 2
    assert graph.m == expected_edges
    assert len(graph.u) == expected_edges
    assert len(graph.v) == expected_edges
    assert len(graph.w) == expected_edges
    assert (graph.u < graph.v).all()
    assert (graph.w >= 0.0).all()
    assert elapsed > 0.0


def test_run_experiment_c():
    rng = np.random.default_rng(42)
    mat = rng.standard_normal((50, 4)).astype(np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    results = run_experiment_c(
        embeddings=mat,
        sample_sizes=[20, 40],
        k=5,
        algo_name="Prim",
        repeats=1,
    )
    assert len(results) == 2
    assert results[0]["sample_size"] == 20
    assert results[0]["complete_edges"] == 20 * 19 // 2
    assert results[0]["knn_edges"] < results[0]["complete_edges"]
    assert results[1]["sample_size"] == 40
    assert results[1]["complete_edges"] == 40 * 39 // 2
