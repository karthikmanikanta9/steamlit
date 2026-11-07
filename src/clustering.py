"""MST edge-cut clustering: size-constrained heuristic (Phase 6).

Algorithm (from the project specification):
  1. Sort MST edges from highest weight to lowest weight.
  2. Consider each candidate in that order.
  3. Remove a candidate only if it splits one current component into two
     and both resulting components contain at least `min_size` documents.
  4. Stop when `target_k` clusters are reached or no eligible edge remains.
  5. If the target cannot be reached, report the actual count and explain why.

For disconnected spanning forests, the initial number of components (trees)
is treated as the starting cluster count. This is reported clearly to the user.
"""
from __future__ import annotations

import numpy as np

from src.knn_graph import Graph
from src.mst import MSTResult


# ---------------------------------------------------------------------------
# Internal graph helpers (tree-only adjacency)
# ---------------------------------------------------------------------------

def _build_adj(n: int, graph: Graph, active_edge_ids: set) -> list[list[int]]:
    """Undirected adjacency list from a subset of edge IDs."""
    adj: list[list[int]] = [[] for _ in range(n)]
    for eid in active_edge_ids:
        u = int(graph.u[eid])
        v = int(graph.v[eid])
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _bfs(adj: list[list[int]], start: int) -> set[int]:
    """Return the set of vertices reachable from *start* via BFS."""
    seen: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        for nb in adj[node]:
            if nb not in seen:
                stack.append(nb)
    return seen


def _assign_labels(n: int, graph: Graph, active_edge_ids: set) -> np.ndarray:
    """Return an integer label array (0-indexed) for every vertex."""
    adj = _build_adj(n, graph, active_edge_ids)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        comp = _bfs(adj, start)
        for node in comp:
            labels[node] = cluster_id
        cluster_id += 1
    return labels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cut_mst_edges(
    graph: Graph,
    mst_result: MSTResult,
    target_k: int = 4,
    min_size: int = 10,
) -> tuple[np.ndarray, list[int], int, dict]:
    """Size-constrained MST edge-cut clustering.

    Parameters
    ----------
    graph       : the kNN Graph the MST was built from
    mst_result  : MST/MSF produced by one of the three algorithms
    target_k    : desired number of clusters (default 4)
    min_size    : minimum documents per resulting component (default 10)

    Returns
    -------
    labels        : np.ndarray shape (n,) — cluster ID per vertex, 0-indexed
    cut_edge_ids  : list[int] — MST edge IDs that were cut
    actual_k      : int — clusters actually produced
    info          : dict — run details for display and logging
    """
    n = graph.n
    active: set[int] = {int(i) for i in mst_result.edge_ids}

    # Initial component count from the spanning forest
    initial_labels = _assign_labels(n, graph, active)
    initial_k = int(initial_labels.max()) + 1 if n > 0 else 0

    cut_edge_ids: list[int] = []
    skipped: list[dict] = []
    current_k = initial_k

    # Sort candidate MST edges: heaviest first; ties broken by larger edge ID first
    candidates = sorted(active, key=lambda i: (-float(graph.w[i]), -i))

    for edge_id in candidates:
        if current_k >= target_k:
            break

        u_node = int(graph.u[edge_id])
        v_node = int(graph.v[edge_id])

        # Try the removal: build adjacency without this edge
        test_active = active - {edge_id}
        adj = _build_adj(n, graph, test_active)

        comp_u = _bfs(adj, u_node)
        # v_node must be in a different component (MST edges are bridges)
        if v_node in comp_u:
            # Edge was not a bridge in the forest (should not happen with valid MST)
            skipped.append({
                "edge_id": edge_id,
                "weight": float(graph.w[edge_id]),
                "reason": "not a bridge (unexpected)",
            })
            continue

        comp_v = _bfs(adj, v_node)
        size_u, size_v = len(comp_u), len(comp_v)

        if size_u >= min_size and size_v >= min_size:
            active = test_active
            cut_edge_ids.append(edge_id)
            current_k += 1
        else:
            skipped.append({
                "edge_id": edge_id,
                "weight": float(graph.w[edge_id]),
                "size_a": size_u,
                "size_b": size_v,
                "reason": (
                    f"sizes {size_u} and {size_v} — "
                    f"both must be ≥ {min_size}"
                ),
            })

    labels = _assign_labels(n, graph, active)
    actual_k = int(labels.max()) + 1 if n > 0 else 0

    info = {
        "target_k": target_k,
        "actual_k": actual_k,
        "min_size": min_size,
        "initial_components": initial_k,
        "cuts_made": len(cut_edge_ids),
        "skipped_count": len(skipped),
        "skipped_details": skipped[:20],
        "reached_target": actual_k >= target_k,
    }
    return labels, cut_edge_ids, actual_k, info


def cluster_sizes(labels: np.ndarray) -> dict[int, int]:
    """Return {cluster_id: document_count} sorted by cluster ID."""
    ids, counts = np.unique(labels, return_counts=True)
    return {int(k): int(v) for k, v in zip(ids, counts)}


def clusters_to_index_map(labels: np.ndarray) -> dict[int, list[int]]:
    """Return {cluster_id: [vertex_indices]} for every cluster."""
    result: dict[int, list[int]] = {}
    for vertex, cid in enumerate(labels.tolist()):
        result.setdefault(int(cid), []).append(vertex)
    return result
