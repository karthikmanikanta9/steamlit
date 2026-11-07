"""Clustering evaluation metrics: ARI, NMI, Silhouette (Phase 7).

Ground-truth labels are ONLY used here, for evaluation.
They must NEVER be fed to clustering algorithms.

Noise points (label == -1, from HDBSCAN) are excluded from every metric.
Singleton clusters are handled gracefully (silhouette returns None).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_noise(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Remove rows where pred_labels == -1 (HDBSCAN noise)."""
    mask = pred_labels != -1
    noise_count = int((~mask).sum())
    tl = true_labels[mask]
    pl = pred_labels[mask]
    mx = matrix[mask] if matrix is not None else None
    return tl, pl, mx, noise_count


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def compute_ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Adjusted Rand Index (noise points excluded)."""
    tl, pl, _, noise = _drop_noise(
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(pred_labels, dtype=np.int64),
    )
    if len(np.unique(pl)) < 2:
        return {
            "ari": None,
            "noise_excluded": noise,
            "note": "Need ≥ 2 predicted clusters for ARI",
        }
    score = float(adjusted_rand_score(tl, pl))
    return {"ari": round(score, 4), "noise_excluded": noise}


def compute_nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> dict:
    """Normalized Mutual Information (noise points excluded)."""
    tl, pl, _, noise = _drop_noise(
        np.asarray(true_labels, dtype=np.int64),
        np.asarray(pred_labels, dtype=np.int64),
    )
    if len(np.unique(pl)) < 2:
        return {
            "nmi": None,
            "noise_excluded": noise,
            "note": "Need ≥ 2 predicted clusters for NMI",
        }
    score = float(normalized_mutual_info_score(tl, pl, average_method="arithmetic"))
    return {"nmi": round(score, 4), "noise_excluded": noise}


def compute_silhouette(
    matrix: np.ndarray,
    pred_labels: np.ndarray,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict:
    """Silhouette score using cosine distance (noise points excluded).

    Subsamples to `sample_size` when the dataset is large to keep runtime
    reasonable (full silhouette is O(n²)).
    """
    tl_dummy = pred_labels  # not used here
    _, pl, mx, noise = _drop_noise(
        pred_labels,          # dummy true labels
        np.asarray(pred_labels, dtype=np.int64),
        np.asarray(matrix, dtype=np.float32),
    )
    n_unique = len(np.unique(pl))
    if n_unique < 2:
        return {
            "silhouette": None,
            "noise_excluded": noise,
            "note": "Need ≥ 2 clusters for silhouette",
        }
    if len(pl) < 2:
        return {
            "silhouette": None,
            "noise_excluded": noise,
            "note": "Too few samples after noise exclusion",
        }

    # Check for singleton clusters (silhouette undefined)
    _, counts = np.unique(pl, return_counts=True)
    if (counts == 1).any():
        # Remove singleton-cluster members
        singleton_ids = set(np.unique(pl)[counts == 1].tolist())
        mask2 = np.array([c not in singleton_ids for c in pl.tolist()])
        pl = pl[mask2]
        mx = mx[mask2]
        if len(np.unique(pl)) < 2 or len(pl) < 2:
            return {
                "silhouette": None,
                "noise_excluded": noise,
                "note": "Singleton clusters prevent silhouette computation",
            }

    # Subsample for speed
    actual_sample = min(len(pl), sample_size)
    try:
        score = float(
            silhouette_score(
                mx,
                pl,
                metric="cosine",
                sample_size=actual_sample,
                random_state=seed,
            )
        )
    except Exception as exc:
        return {"silhouette": None, "noise_excluded": noise, "note": str(exc)}

    return {
        "silhouette": round(score, 4),
        "noise_excluded": noise,
        "sample_used": actual_sample,
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

def evaluate_all(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    matrix: np.ndarray,
    sample_size: int = 2000,
) -> dict:
    """Compute ARI, NMI, and silhouette in a single call.

    Returns a flat dict suitable for logging and display.
    """
    true_labels = np.asarray(true_labels, dtype=np.int64)
    pred_labels = np.asarray(pred_labels, dtype=np.int64)
    matrix = np.asarray(matrix, dtype=np.float32)

    ari_res = compute_ari(true_labels, pred_labels)
    nmi_res = compute_nmi(true_labels, pred_labels)
    sil_res = compute_silhouette(matrix, pred_labels, sample_size=sample_size)

    notes = [
        v for d in (ari_res, nmi_res, sil_res) for k, v in d.items() if k == "note"
    ]
    return {
        "ari": ari_res.get("ari"),
        "nmi": nmi_res.get("nmi"),
        "silhouette": sil_res.get("silhouette"),
        "noise_excluded": max(
            ari_res.get("noise_excluded", 0),
            nmi_res.get("noise_excluded", 0),
        ),
        "notes": notes,
    }
