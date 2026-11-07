"""Exact cosine kNN graph with union symmetrization (Phase 4)."""
import json
import time
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src import config


# ---------- Graph container ----------
@dataclass(frozen=True, eq=False)
class Graph:
    """Undirected weighted graph stored as edge arrays (read-only).

    Edge number i joins vertices u[i] and v[i] (u < v) with weight w[i].
    The edge index i is the stable edge ID used for deterministic tie-breaking.
    """
    n: int
    u: np.ndarray
    v: np.ndarray
    w: np.ndarray

    def __post_init__(self):
        for array in (self.u, self.v, self.w):
            array.flags.writeable = False

    @property
    def m(self) -> int:
        return int(len(self.u))


def make_graph(n, u, v, w) -> Graph:
    """Build a Graph from edge lists (copies the data, stores u < v)."""
    u = np.array(u, dtype=np.int64)
    v = np.array(v, dtype=np.int64)
    w = np.array(w, dtype=np.float64)
    return Graph(int(n), np.minimum(u, v), np.maximum(u, v), w)


# ---------- Neighbour search ----------
def knn_search(embeddings, k, chunk_size=config.KNN_CHUNK_SIZE) -> np.ndarray:
    """Exact k nearest neighbours by cosine similarity (unit-length vectors).

    Returns an (n, k) array: row i lists i's neighbours, nearest first.
    A document is never its own neighbour (excluded by index, so exact
    duplicates of a document can still be neighbours).
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    n = emb.shape[0]
    if not 1 <= k < n:
        raise ValueError(f"k must be between 1 and {n - 1}, got {k}")

    neighbours = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sims = emb[start:stop] @ emb.T
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        part_sims = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_sims, axis=1, kind="stable")
        neighbours[start:stop] = np.take_along_axis(part, order, axis=1)
    return neighbours


def symmetrize_union(n, neighbours):
    """Union symmetrization: keep edge {i, j} if i chose j OR j chose i.

    Returns (u, v, info) with u < v, no self-loops, no duplicates,
    sorted by (u, v) so edge IDs are deterministic.
    """
    k = neighbours.shape[1]
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = neighbours.ravel().astype(np.int64)
    low, high = np.minimum(src, dst), np.maximum(src, dst)

    not_loop = low != high
    self_loops = int((~not_loop).sum())
    low, high = low[not_loop], high[not_loop]

    keys = np.unique(low * n + high)
    u, v = keys // n, keys % n
    info = {
        "directed_selections": int(n * k),
        "self_loops_removed": self_loops,
        "mutual_pairs_merged": int(len(low) - len(keys)),
    }
    return u, v, info


def edge_weights(embeddings, u, v, chunk=10000) -> np.ndarray:
    """Cosine distance 1 - cos for each edge, computed once in float64.

    Every undirected pair gets exactly one weight, whichever side chose it.
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    weights = np.empty(len(u), dtype=np.float64)
    for start in range(0, len(u), chunk):
        stop = start + chunk
        dots = np.einsum("ij,ij->i", emb[u[start:stop]], emb[v[start:stop]])
        weights[start:stop] = dots
    return np.clip(1.0 - weights, 0.0, 2.0)


# ---------- Statistics ----------
def connected_component_labels(graph: Graph):
    """Return (component_count, label_per_vertex)."""
    adjacency = csr_matrix((np.ones(graph.m), (graph.u, graph.v)), shape=(graph.n, graph.n))
    count, labels = connected_components(adjacency, directed=False)
    return int(count), labels


def graph_statistics(graph: Graph) -> dict:
    count, labels = connected_component_labels(graph)
    sizes = np.sort(np.bincount(labels))[::-1]
    degrees = np.bincount(graph.u, minlength=graph.n) + np.bincount(graph.v, minlength=graph.n)
    pairs = graph.n * (graph.n - 1) / 2
    return {
        "vertices": int(graph.n),
        "edges": graph.m,
        "density": float(graph.m / pairs) if pairs else 0.0,
        "average_degree": float(2 * graph.m / graph.n),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "components": count,
        "connected": count == 1,
        "largest_component": int(sizes[0]),
        "component_sizes": [int(s) for s in sizes[:50]],
        "weight_min": float(graph.w.min()) if graph.m else 0.0,
        "weight_mean": float(graph.w.mean()) if graph.m else 0.0,
        "weight_max": float(graph.w.max()) if graph.m else 0.0,
    }


# ---------- Full construction ----------
def build_knn_graph(embeddings, k):
    """Build the undirected kNN graph. Returns (Graph, statistics)."""
    n = len(embeddings)

    started = time.perf_counter()
    neighbours = knn_search(embeddings, k)
    search_seconds = time.perf_counter() - started

    started = time.perf_counter()
    u, v, info = symmetrize_union(n, neighbours)
    w = edge_weights(embeddings, u, v)
    graph = Graph(n, u, v, w)
    edge_seconds = time.perf_counter() - started

    started = time.perf_counter()
    stats = graph_statistics(graph)
    stats_seconds = time.perf_counter() - started

    stats.update(info)
    stats.update({
        "k": int(k),
        "search_seconds": round(search_seconds, 4),
        "edge_build_seconds": round(edge_seconds, 4),
        "build_seconds": round(search_seconds + edge_seconds, 4),
        "statistics_seconds": round(stats_seconds, 4),
    })
    return graph, stats


# ---------- Saving and loading ----------
def embeddings_source_key(embedding_meta) -> str:
    """Identifies which embeddings a graph was built from."""
    return f"{embedding_meta['dataset_fingerprint']}|{embedding_meta['created_utc']}"


def _graph_paths(k):
    return (config.GRAPHS_DIR / f"knn_k{k}.npz", config.GRAPHS_DIR / f"knn_k{k}_meta.json")


def save_graph(graph: Graph, stats: dict, source_key: str) -> None:
    npz_path, meta_path = _graph_paths(stats["k"])
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, u=graph.u, v=graph.v, w=graph.w, n=np.int64(graph.n))
    with open(meta_path, "w", encoding="utf-8") as file:
        json.dump({"source_key": source_key, "stats": stats}, file, indent=2)


def load_graph(k, source_key):
    """Return (Graph, stats) if a matching saved graph exists, else None."""
    npz_path, meta_path = _graph_paths(k)
    if not (npz_path.exists() and meta_path.exists()):
        return None
    with open(meta_path, encoding="utf-8") as file:
        saved = json.load(file)
    if saved.get("source_key") != source_key:
        return None
    with np.load(npz_path, allow_pickle=False) as data:
        graph = Graph(int(data["n"]), data["u"].astype(np.int64),
                      data["v"].astype(np.int64), data["w"].astype(np.float64))
    return graph, saved["stats"]


# ---------- Exploring neighbourhoods ----------
def neighbour_lists(graph: Graph):
    """adjacency[i] = list of (weight, neighbour), nearest first."""
    adjacency = [[] for _ in range(graph.n)]
    for a, b, weight in zip(graph.u.tolist(), graph.v.tolist(), graph.w.tolist()):
        adjacency[a].append((weight, b))
        adjacency[b].append((weight, a))
    for row in adjacency:
        row.sort()
    return adjacency


def bfs_neighbourhood(adjacency, centre, hops, max_nodes):
    """Breadth-first nodes around `centre`. Returns (nodes, depth_by_node)."""
    nodes = [centre]
    depth = {centre: 0}
    frontier = [centre]
    for hop in range(1, hops + 1):
        next_frontier = []
        for node in frontier:
            for _, neighbour in adjacency[node]:
                if neighbour in depth:
                    continue
                if len(nodes) >= max_nodes:
                    return nodes, depth
                depth[neighbour] = hop
                nodes.append(neighbour)
                next_frontier.append(neighbour)
        frontier = next_frontier
    return nodes, depth


@dataclass(frozen=True, eq=False)
class SubGraph:
    graph: Graph            # vertices renumbered 0..len(nodes)-1
    nodes: np.ndarray       # original vertex of each local vertex
    original_edge_ids: np.ndarray


def induced_subgraph(graph: Graph, nodes) -> SubGraph:
    """Graph on `nodes` with every edge whose two ends are both in `nodes`."""
    nodes = np.asarray(nodes, dtype=np.int64)
    local = np.full(graph.n, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    keep = (local[graph.u] >= 0) & (local[graph.v] >= 0)
    ids = np.flatnonzero(keep)
    sub = make_graph(len(nodes), local[graph.u[ids]], local[graph.v[ids]], graph.w[ids])
    return SubGraph(sub, nodes, ids)


def saved_graph_ks(source_key):
    """Which k values have a saved graph built from these embeddings (cheap check)."""
    found = []
    for k in config.K_VALUES:
        _, meta_path = _graph_paths(k)
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as file:
                if json.load(file).get("source_key") == source_key:
                    found.append(k)
    return found