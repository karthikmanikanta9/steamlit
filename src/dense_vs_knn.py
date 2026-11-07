"""Experiment C: Complete Graph vs. Sparse kNN Graph (Phase 7 / 8).

Compares:
  - Complete graph: all n(n-1)/2 pairs explicitly materialized
  - Sparse kNN graph: union-symmetrized kNN graph with parameter k

Evaluates edge count, graph construction time, MST execution time, and memory.
"""
from __future__ import annotations

import gc
import time
import tracemalloc
import numpy as np

from src.knn_graph import Graph, make_graph, build_knn_graph
from src.mst import prim, kruskal, boruvka
from src.benchmark import benchmark_algorithm


def build_complete_graph(embeddings: np.ndarray, chunk_size: int = 500000) -> tuple[Graph, float]:
    """Materialize the complete graph on embeddings with pairwise cosine distance.

    Returns (graph, construction_seconds).
    """
    n = len(embeddings)
    t0 = time.perf_counter()

    u_indices, v_indices = np.triu_indices(n, k=1)
    total_edges = len(u_indices)
    weights = np.empty(total_edges, dtype=np.float64)

    emb = np.asarray(embeddings, dtype=np.float64)
    for start in range(0, total_edges, chunk_size):
        end = min(start + chunk_size, total_edges)
        sub_u = u_indices[start:end]
        sub_v = v_indices[start:end]
        dots = np.einsum("ij,ij->i", emb[sub_u], emb[sub_v])
        weights[start:end] = np.clip(1.0 - dots, 0.0, 2.0)

    graph = make_graph(n, u_indices, v_indices, weights)
    elapsed = time.perf_counter() - t0
    return graph, elapsed


def run_experiment_c(
    embeddings: np.ndarray,
    sample_sizes: list[int] = (500, 1000, 1500, 2000),
    k: int = 10,
    algo_name: str = "Prim",
    repeats: int = 3,
    progress_callback=None,
) -> list[dict]:
    """Run Experiment C comparing Complete Graph vs kNN Graph across sample sizes.

    Returns a list of dict comparison records.
    """
    from src.mst import ALGORITHMS

    algo_fn = ALGORITHMS.get(algo_name, prim)
    results = []
    total_steps = len(sample_sizes) * 2
    step = 0

    for sample_size in sample_sizes:
        if sample_size > len(embeddings):
            continue

        sub_emb = embeddings[:sample_size]

        # 1. Complete Graph
        if progress_callback:
            progress_callback(step, total_steps, f"Building complete graph for n = {sample_size:,}...")
        complete_graph, complete_build_sec = build_complete_graph(sub_emb)
        step += 1

        if progress_callback:
            progress_callback(step, total_steps, f"Running {algo_name} on complete graph (n = {sample_size:,})...")
        complete_bench = benchmark_algorithm(algo_name, algo_fn, complete_graph, repeats=repeats)
        step += 1

        # 2. kNN Graph
        knn_graph, knn_stats = build_knn_graph(sub_emb, k)
        knn_bench = benchmark_algorithm(algo_name, algo_fn, knn_graph, repeats=repeats)

        theoretical_complete_edges = sample_size * (sample_size - 1) // 2

        results.append({
            "sample_size": sample_size,
            "k": k,
            "algorithm": algo_name,
            # Complete graph metrics
            "complete_edges": theoretical_complete_edges,
            "complete_build_seconds": round(complete_build_sec, 4),
            "complete_mst_seconds": round(complete_bench.median_seconds, 4),
            "complete_mst_memory_mb": round(complete_bench.peak_memory_bytes / 1e6, 3),
            "complete_mst_weight": round(complete_bench.result.total_weight, 6),
            # kNN graph metrics
            "knn_edges": knn_graph.m,
            "knn_build_seconds": round(knn_stats["build_seconds"], 4),
            "knn_mst_seconds": round(knn_bench.median_seconds, 4),
            "knn_mst_memory_mb": round(knn_bench.peak_memory_bytes / 1e6, 3),
            "knn_mst_weight": round(knn_bench.result.total_weight, 6),
            # Comparisons / speedup / reduction
            "edge_reduction_ratio": round(theoretical_complete_edges / max(1, knn_graph.m), 1),
            "mst_speedup": round(complete_bench.median_seconds / max(1e-6, knn_bench.median_seconds), 2),
        })

    return results
