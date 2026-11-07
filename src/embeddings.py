"""Generate, cache and inspect sentence embeddings (Phase 3)."""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.data_loader import subset_fingerprint


# ---------- Generation ----------
def load_model(model_id=config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE):
    """Load the sentence-transformer model (downloads about 90 MB the first time)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=device)


def l2_normalize(matrix) -> np.ndarray:
    """Scale every row to unit length (all-zero rows stay zero)."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_texts(texts, model, batch_size=config.EMBEDDING_BATCH_SIZE, progress_callback=None):
    """Encode texts in batches and return a unit-length float32 matrix."""
    texts = list(texts)
    total = len(texts)
    chunks = []
    for begin in range(0, total, batch_size):
        batch = texts[begin:begin + batch_size]
        vectors = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        chunks.append(np.asarray(vectors, dtype=np.float32))
        if progress_callback is not None:
            progress_callback(min(begin + batch_size, total), total)
    return l2_normalize(np.vstack(chunks))


def _library_versions() -> dict:
    versions = {}
    for name in ("sentence_transformers", "torch", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def create_and_save_embeddings(df: pd.DataFrame, progress_callback=None, model=None):
    """Encode every document, save matrix + ID mapping + metadata, return (matrix, meta)."""
    started = time.perf_counter()
    if model is None:
        model = load_model()
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    matrix = encode_texts(df["text"].tolist(), model, config.EMBEDDING_BATCH_SIZE, progress_callback)
    encode_seconds = time.perf_counter() - started

    max_len = getattr(model, "max_seq_length", None)
    meta = {
        "model": config.EMBEDDING_MODEL,
        "device": config.EMBEDDING_DEVICE,
        "num_documents": int(matrix.shape[0]),
        "dimension": int(matrix.shape[1]),
        "normalization": "L2 (unit-length vectors)",
        "dtype": "float32",
        "batch_size": config.EMBEDDING_BATCH_SIZE,
        "max_seq_length": int(max_len) if max_len is not None else None,
        "dataset_fingerprint": subset_fingerprint(df),
        "model_load_seconds": round(load_seconds, 2),
        "encode_seconds": round(encode_seconds, 2),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_versions": _library_versions(),
    }
    save_embeddings(matrix, df["doc_id"].tolist(), meta)
    return matrix, meta


# ---------- Saving and loading ----------
def save_embeddings(matrix, doc_ids, meta) -> None:
    config.EMBEDDINGS_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.EMBEDDINGS_NPZ,
        embeddings=np.asarray(matrix, dtype=np.float32),
        doc_ids=np.array(list(doc_ids), dtype=str),
    )
    with open(config.EMBEDDINGS_META, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)


def read_embeddings_meta():
    """Return the saved embedding metadata, or None (cheap check for other pages)."""
    if not config.EMBEDDINGS_META.exists():
        return None
    with open(config.EMBEDDINGS_META, encoding="utf-8") as file:
        return json.load(file)


def load_cached_embeddings(df: pd.DataFrame):
    """Load saved embeddings if they match the current dataset and model.

    Returns (matrix, meta, message). matrix is None when there is no usable cache.
    """
    if not (config.EMBEDDINGS_NPZ.exists() and config.EMBEDDINGS_META.exists()):
        return None, None, "No cached embeddings found yet."

    meta = read_embeddings_meta()
    with np.load(config.EMBEDDINGS_NPZ, allow_pickle=False) as data:
        matrix = data["embeddings"]
        saved_ids = [str(item) for item in data["doc_ids"]]

    if meta.get("model") != config.EMBEDDING_MODEL:
        return None, None, "Cache ignored: it was made with a different model."
    if meta.get("dataset_fingerprint") != subset_fingerprint(df):
        return None, None, "Cache ignored: the dataset subset has changed since it was made."
    if saved_ids != df["doc_id"].tolist():
        return None, None, "Cache ignored: document IDs do not match the current subset."
    if matrix.shape != (len(df), meta.get("dimension")):
        return None, None, "Cache ignored: the matrix shape is not what was expected."
    return matrix, meta, "Loaded from cache."


# ---------- Checks and similarity ----------
def validate_embeddings(matrix, expected_rows):
    """Return a list of (check name, passed, detail)."""
    matrix = np.asarray(matrix)
    checks = [("One vector per document", matrix.shape[0] == expected_rows,
               f"{matrix.shape[0]} vectors, {expected_rows} documents")]
    checks.append(("Values are finite (no NaN or infinity)", bool(np.isfinite(matrix).all()), ""))
    deviation = float(np.abs(np.linalg.norm(matrix, axis=1) - 1.0).max())
    checks.append(("Vectors have unit length", deviation < 1e-4, f"largest deviation {deviation:.2e}"))
    checks.append(("Stored as float32", matrix.dtype == np.float32, str(matrix.dtype)))
    return checks


def cosine_distance(similarity):
    """cosine distance = 1 - cosine similarity (clipped to the valid range 0..2)."""
    return np.clip(1.0 - np.asarray(similarity, dtype=np.float64), 0.0, 2.0)


def most_similar(matrix, index, top=5):
    """Indices and cosine similarities of the `top` documents most similar to `index`.

    Vectors are unit length, so cosine similarity is just the dot product.
    """
    similarities = matrix @ matrix[index]
    ranked = similarities.copy()
    ranked[index] = -np.inf
    order = np.argsort(-ranked, kind="stable")[:top]
    return order, similarities[order]