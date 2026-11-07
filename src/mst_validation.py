"""Independent checks for MST/forest results (Phase 5)."""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from src import config
from src.knn_graph import Graph, connected_component_labels
from src.mst import DisjointSet, MSTResult


def scipy_reference_weight(graph: Graph) -> float:
    """Total MST/forest weight from SciPy, used ONLY as an independent check.

    SciPy treats a stored 0 as 'no edge', so 1 is added to every weight and
    subtracted again (every spanning forest has the same number of edges).
    """
    shifted = csr_matrix((graph.w + 1.0, (graph.u, graph.v)), shape=(graph.n, graph.n))
    forest = minimum_spanning_tree(shifted)
    components, _ = connected_component_labels(graph)
    return float(forest.sum()) - (graph.n - components)


def _close(a, b, tol=config.WEIGHT_TOLERANCE):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def validate_mst(graph: Graph, result: MSTResult, reference_weight=None):
    """Return a list of (check name, passed, detail)."""
    components, _ = connected_component_labels(graph)
    ids = result.edge_ids
    checks = []

    ids_ok = bool(len(ids) == 0 or (ids.min() >= 0 and ids.max() < graph.m))
    unique = len(set(ids.tolist())) == len(ids)
    checks.append(("Edge IDs are valid and unique", ids_ok and unique, ""))

    expected = graph.n - components
    checks.append((f"Edge count equals V - C = {graph.n} - {components}",
                   len(ids) == expected, f"{len(ids)} edges, expected {expected}"))

    if ids_ok:
        dsu = DisjointSet(graph.n)
        acyclic = all(dsu.union(int(graph.u[i]), int(graph.v[i])) for i in ids)
        checks.append(("No cycles", acyclic, ""))
        checks.append(("Spans every component of the graph", dsu.components == components,
                       f"{dsu.components} trees, graph has {components} components"))
        recomputed = math.fsum(float(graph.w[i]) for i in ids)
        checks.append(("Reported total weight matches the chosen edges",
                       _close(recomputed, result.total_weight), f"{recomputed:.10f}"))
    else:
        checks.append(("No cycles", False, "skipped: invalid edge IDs"))

    if reference_weight is not None:
        checks.append(("Total weight matches the SciPy reference",
                       _close(result.total_weight, reference_weight),
                       f"{result.total_weight:.10f} vs {reference_weight:.10f}"))
    return checks


def all_passed(checks) -> bool:
    return all(passed for _, passed, _ in checks)


def compare_results(results: dict, tol=config.WEIGHT_TOLERANCE) -> dict:
    """Do the algorithms agree on total weight (and on the exact edge set)?"""
    weights = [r.total_weight for r in results.values()]
    spread = max(weights) - min(weights) if weights else 0.0
    edge_sets = {frozenset(r.edge_ids.tolist()) for r in results.values()}
    return {
        "weights_agree": bool(spread <= tol * max(1.0, max(map(abs, weights), default=1.0))),
        "max_weight_difference": float(spread),
        "identical_edge_sets": len(edge_sets) <= 1,
    }