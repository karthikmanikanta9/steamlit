"""K-means and HDBSCAN baselines (Phase 6/7).

Both methods operate DIRECTLY on the embedding matrix.
They do NOT use the kNN graph or MST built in earlier phases.

Baseline path:  embeddings  →  K-means  (or HDBSCAN)  →  cluster labels
MST path:       embeddings  →  kNN graph  →  MST  →  edge cut  →  labels
"""
from __future__ import annotations

import gc
import time
import tracemalloc

import numpy as np

from src import config


# ---------------------------------------------------------------------------
# K-means
# ---------------------------------------------------------------------------

def run_kmeans(
    matrix: np.ndarray,
    n_clusters: int = config.DEFAULT_N_CLUSTERS,
    seed: int = config.KMEANS_SEED,
    n_init: int = config.KMEANS_N_INIT,
) -> tuple[np.ndarray, dict]:
    """Fit K-means directly on *matrix*. Returns (labels, info)."""
    from sklearn.cluster import KMeans

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=n_init)
    km.fit(matrix)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    labels = km.labels_.astype(np.int64)
    ids, counts = np.unique(labels, return_counts=True)

    info = {
        "method": "K-means",
        "n_clusters_requested": int(n_clusters),
        "n_clusters_actual": int(len(ids)),
        "seed": seed,
        "n_init": n_init,
        "inertia": float(km.inertia_),
        "runtime_seconds": round(elapsed, 4),
        "peak_memory_bytes": int(peak_bytes),
        "cluster_sizes": {int(k): int(v) for k, v in zip(ids, counts)},
        "noise_points": 0,
    }
    return labels, info


# ---------------------------------------------------------------------------
# HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(
    matrix: np.ndarray,
    min_cluster_size: int = config.HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int | None = config.HDBSCAN_MIN_SAMPLES,
) -> tuple[np.ndarray, dict]:
    """Fit HDBSCAN directly on *matrix*. Returns (labels, info).

    Noise points are labelled -1.  They are excluded from ARI/NMI/silhouette.
    Uses sklearn.cluster.HDBSCAN (scikit-learn >= 1.3).
    """
    try:
        from sklearn.cluster import HDBSCAN
        _backend = "sklearn"
    except ImportError:
        try:
            import hdbscan as _hdbscan_pkg
            HDBSCAN = _hdbscan_pkg.HDBSCAN  # type: ignore[assignment]
            _backend = "hdbscan-package"
        except ImportError as err:
            raise ImportError(
                "HDBSCAN is not available. "
                "Install scikit-learn >= 1.3 or run: pip install hdbscan"
            ) from err

    kwargs: dict = {"min_cluster_size": min_cluster_size}
    if min_samples is not None:
        kwargs["min_samples"] = min_samples

    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()

    clusterer = HDBSCAN(**kwargs)
    clusterer.fit(matrix)

    elapsed = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    labels: np.ndarray = np.asarray(clusterer.labels_, dtype=np.int64)
    noise = int((labels == -1).sum())
    non_noise = labels[labels >= 0]

    if len(non_noise) > 0:
        ids, counts = np.unique(non_noise, return_counts=True)
        n_clusters = int(ids.max()) + 1
        sizes = {int(k): int(v) for k, v in zip(ids, counts)}
    else:
        n_clusters = 0
        sizes = {}

    info = {
        "method": "HDBSCAN",
        "backend": _backend,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "runtime_seconds": round(elapsed, 4),
        "peak_memory_bytes": int(peak_bytes),
        "cluster_sizes": sizes,
        "noise_points": noise,
        "noise_fraction": round(noise / len(labels), 4) if len(labels) > 0 else 0.0,
    }
    return labels, info
