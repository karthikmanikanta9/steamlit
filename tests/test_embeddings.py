import json
import zlib

import numpy as np
import pandas as pd
import pytest

from src import config
from src import embeddings as emb


class DummyModel:
    """Fake model: the vector depends only on the text, so batching cannot change it."""
    max_seq_length = 128

    def encode(self, sentences, **kwargs):
        rows = []
        for sentence in sentences:
            rng = np.random.default_rng(zlib.crc32(sentence.encode("utf-8")))
            rows.append(rng.normal(size=8) * 3.0)
        return np.array(rows)


@pytest.fixture
def df():
    return pd.DataFrame({
        "doc_id": [f"d{i}" for i in range(10)],
        "text": [f"text number {i}" for i in range(10)],
        "label": [i % 4 for i in range(10)],
        "label_name": [config.LABEL_NAMES[i % 4] for i in range(10)],
    })


@pytest.fixture(autouse=True)
def temp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "EMBEDDINGS_NPZ", tmp_path / "embeddings.npz")
    monkeypatch.setattr(config, "EMBEDDINGS_META", tmp_path / "embeddings_meta.json")


def test_l2_normalize_gives_unit_rows_and_keeps_zero_rows():
    out = emb.l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])
    assert out.dtype == np.float32


def test_batching_does_not_change_result_and_progress_reaches_total():
    texts = [f"t{i}" for i in range(10)]
    calls = []
    small = emb.encode_texts(texts, DummyModel(), batch_size=3, progress_callback=lambda d, t: calls.append((d, t)))
    big = emb.encode_texts(texts, DummyModel(), batch_size=100)
    assert np.allclose(small, big)
    assert calls[-1] == (10, 10)


def test_save_and_reload_round_trip(df):
    matrix, meta = emb.create_and_save_embeddings(df, model=DummyModel())
    assert matrix.shape == (10, 8)
    assert meta["dimension"] == 8 and meta["num_documents"] == 10
    loaded, loaded_meta, message = emb.load_cached_embeddings(df)
    assert loaded is not None and message == "Loaded from cache."
    assert np.allclose(loaded, matrix)
    assert loaded_meta["dataset_fingerprint"] == meta["dataset_fingerprint"]
    assert all(ok for _, ok, _ in emb.validate_embeddings(loaded, len(df)))


def test_missing_cache_is_reported(df):
    matrix, meta, message = emb.load_cached_embeddings(df)
    assert matrix is None and "No cached" in message


def test_cache_is_rejected_when_dataset_changes(df):
    emb.create_and_save_embeddings(df, model=DummyModel())
    changed = df.copy()
    changed.loc[0, "text"] = "something different"
    assert emb.load_cached_embeddings(changed)[0] is None
    reordered = df.iloc[::-1].reset_index(drop=True)
    assert emb.load_cached_embeddings(reordered)[0] is None


def test_cache_is_rejected_when_model_differs(df):
    emb.create_and_save_embeddings(df, model=DummyModel())
    meta = json.loads(config.EMBEDDINGS_META.read_text())
    meta["model"] = "some/other-model"
    config.EMBEDDINGS_META.write_text(json.dumps(meta))
    assert emb.load_cached_embeddings(df)[0] is None


def test_most_similar_excludes_itself_and_is_sorted():
    matrix = emb.l2_normalize(np.array([[1, 0], [0.9, 0.1], [0, 1], [-1, 0]], dtype=np.float32))
    order, sims = emb.most_similar(matrix, 0, top=3)
    assert 0 not in order.tolist()
    assert order.tolist() == [1, 2, 3]
    assert sims[0] > sims[1] > sims[2]


def test_cosine_distance_values():
    assert emb.cosine_distance(1.0) == 0.0
    assert emb.cosine_distance(0.0) == 1.0
    assert emb.cosine_distance(-1.0) == 2.0
    assert emb.cosine_distance(1.0000001) == 0.0   # rounding noise is clipped