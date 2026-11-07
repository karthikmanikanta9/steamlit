"""Tests for clustering evaluation metrics (Phase 7).

Uses small synthetic label arrays with known expected values.
"""
import numpy as np
import pytest

from src.evaluation import compute_ari, compute_nmi, compute_silhouette, evaluate_all


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _perfect_labels(n_per_class=10, n_classes=4):
    """Perfect agreement: true == pred (ARI = NMI = 1.0)."""
    labels = np.repeat(np.arange(n_classes), n_per_class)
    return labels.copy(), labels.copy()


def _random_embeddings(n, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / norms


# ---------------------------------------------------------------------------
# ARI
# ---------------------------------------------------------------------------

class TestARI:
    def test_perfect_agreement(self):
        true, pred = _perfect_labels()
        result = compute_ari(true, pred)
        assert result["ari"] == pytest.approx(1.0, abs=1e-4)

    def test_random_labels_low_ari(self):
        rng = np.random.default_rng(42)
        true = np.repeat(np.arange(4), 25)
        pred = rng.integers(0, 4, size=100)
        result = compute_ari(true, pred)
        assert result["ari"] is not None
        assert result["ari"] < 0.5

    def test_noise_exclusion(self):
        true = np.array([0, 0, 1, 1, 0])
        pred = np.array([0, 0, 1, 1, -1])    # last point is noise
        result = compute_ari(true, pred)
        assert result["noise_excluded"] == 1
        assert result["ari"] == pytest.approx(1.0, abs=1e-4)

    def test_single_cluster_returns_none(self):
        true = np.array([0, 1, 0, 1])
        pred = np.array([0, 0, 0, 0])        # all in one cluster
        result = compute_ari(true, pred)
        assert result["ari"] is None

    def test_all_noise_returns_none(self):
        true = np.array([0, 1, 0])
        pred = np.array([-1, -1, -1])
        result = compute_ari(true, pred)
        assert result["ari"] is None


# ---------------------------------------------------------------------------
# NMI
# ---------------------------------------------------------------------------

class TestNMI:
    def test_perfect_agreement(self):
        true, pred = _perfect_labels()
        result = compute_nmi(true, pred)
        assert result["nmi"] == pytest.approx(1.0, abs=1e-4)

    def test_range_zero_to_one(self):
        rng = np.random.default_rng(7)
        true = np.repeat(np.arange(4), 25)
        pred = rng.integers(0, 4, size=100)
        result = compute_nmi(true, pred)
        assert result["nmi"] is not None
        assert 0.0 <= result["nmi"] <= 1.0

    def test_noise_exclusion(self):
        true = np.array([0, 0, 1, 1, 0])
        pred = np.array([0, 0, 1, 1, -1])
        result = compute_nmi(true, pred)
        assert result["noise_excluded"] == 1
        assert result["nmi"] == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------

class TestSilhouette:
    def test_well_separated_clusters(self):
        # Two tight, well-separated clusters in 2-D embedding space
        rng = np.random.default_rng(0)
        mat_a = rng.normal([1, 0], 0.05, (20, 2)).astype(np.float32)
        mat_b = rng.normal([-1, 0], 0.05, (20, 2)).astype(np.float32)
        mat = np.vstack([mat_a, mat_b])
        # L2-normalise rows
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / norms
        pred = np.array([0] * 20 + [1] * 20)
        result = compute_silhouette(mat, pred, sample_size=40)
        assert result["silhouette"] is not None
        assert result["silhouette"] > 0.0

    def test_single_cluster_returns_none(self):
        mat = _random_embeddings(20)
        pred = np.zeros(20, dtype=np.int64)
        result = compute_silhouette(mat, pred)
        assert result["silhouette"] is None

    def test_noise_excluded(self):
        mat = _random_embeddings(20)
        pred = np.array([0] * 10 + [1] * 9 + [-1])
        result = compute_silhouette(mat, pred)
        assert result["noise_excluded"] == 1


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_perfect_returns_ones(self):
        true, pred = _perfect_labels(n_per_class=10)
        mat = _random_embeddings(40)
        result = evaluate_all(true, pred, mat)
        assert result["ari"] == pytest.approx(1.0, abs=1e-4)
        assert result["nmi"] == pytest.approx(1.0, abs=1e-4)

    def test_returns_all_keys(self):
        true, pred = _perfect_labels(n_per_class=5)
        mat = _random_embeddings(20)
        result = evaluate_all(true, pred, mat)
        for key in ("ari", "nmi", "silhouette", "noise_excluded", "notes"):
            assert key in result
