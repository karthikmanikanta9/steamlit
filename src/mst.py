"""Prim, Kruskal and Boruvka, written from scratch (Phase 5).

Every algorithm takes the same read-only Graph and returns an MSTResult.
Edges are compared by the key (weight, edge_id). Because edge IDs are unique,
that is a strict total order, so the minimum spanning forest is unique and all
three algorithms must select exactly the same edges.
"""
import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.knn_graph import Graph


class DisjointSet:
    """Union-Find with path compression and union by size."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:            # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.size[root_a] < self.size[root_b]:  # union by size
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.components -= 1
        return True


@dataclass
class MSTResult:
    algorithm: str
    edge_ids: np.ndarray                 # in the order the edges were accepted
    total_weight: float
    num_vertices: int
    phases: Optional[int] = None         # Boruvka only
    phase_ends: Optional[list] = None    # Boruvka only: edges accepted after each phase

    @property
    def num_edges(self) -> int:
        return int(len(self.edge_ids))


def _finish(name, graph, selected, **extra):
    w = graph.w
    total = math.fsum(float(w[i]) for i in selected)
    return MSTResult(name, np.array(selected, dtype=np.int64), total, graph.n, **extra)


# ---------- Kruskal ----------
def kruskal(graph: Graph) -> MSTResult:
    """Sort edges, add an edge if it joins two different components."""
    n = graph.n
    u, v, w = graph.u.tolist(), graph.v.tolist(), graph.w.tolist()
    order = sorted(range(len(w)), key=w.__getitem__)   # stable sort: ties keep edge-ID order
    dsu = DisjointSet(n)
    selected = []
    for i in order:
        if dsu.union(u[i], v[i]):
            selected.append(i)
            if len(selected) == n - 1:
                break
    return _finish("Kruskal", graph, selected)


# ---------- Prim ----------
def prim(graph: Graph, start: int = 0) -> MSTResult:
    """Grow a tree from `start` with a binary heap; restart in each new component."""
    n = graph.n
    u, v, w = graph.u.tolist(), graph.v.tolist(), graph.w.tolist()
    adjacency = [[] for _ in range(n)]
    for i in range(len(w)):
        adjacency[u[i]].append((w[i], i, v[i]))
        adjacency[v[i]].append((w[i], i, u[i]))

    visited = [False] * n
    selected = []
    roots = [start] + [x for x in range(n) if x != start]
    for root in roots:
        if visited[root]:
            continue
        visited[root] = True
        heap = list(adjacency[root])
        heapq.heapify(heap)
        while heap:
            weight, edge_id, vertex = heapq.heappop(heap)   # (weight, id) order
            if visited[vertex]:
                continue
            visited[vertex] = True
            selected.append(edge_id)
            for entry in adjacency[vertex]:
                if not visited[entry[2]]:
                    heapq.heappush(heap, entry)
    return _finish("Prim", graph, selected)


# ---------- Boruvka ----------
def boruvka(graph: Graph) -> MSTResult:
    """Every component picks its cheapest outgoing edge; merge; repeat."""
    n = graph.n
    u, v, w = graph.u.tolist(), graph.v.tolist(), graph.w.tolist()
    dsu = DisjointSet(n)
    active = list(range(len(w)))      # edges that may still join two components
    selected, phase_ends = [], []

    while active:
        cheapest = [-1] * n           # indexed by component root
        for i in active:
            root_u, root_v = dsu.find(u[i]), dsu.find(v[i])
            if root_u == root_v:
                continue
            for root in (root_u, root_v):
                best = cheapest[root]
                if best == -1 or (w[i], i) < (w[best], best):
                    cheapest[root] = i

        merged = False
        for i in cheapest:
            if i != -1 and dsu.union(u[i], v[i]):
                selected.append(i)
                merged = True
        if not merged:
            break
        phase_ends.append(len(selected))
        active = [i for i in active if dsu.find(u[i]) != dsu.find(v[i])]

    return _finish("Borůvka", graph, selected, phases=len(phase_ends), phase_ends=phase_ends)


ALGORITHMS = {"Prim": prim, "Kruskal": kruskal, "Borůvka": boruvka}