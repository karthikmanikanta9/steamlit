"""Shared helpers for Streamlit pages: get the dataset, embeddings and graphs."""
import streamlit as st

from src import config
from src.data_loader import load_saved_subset
from src.embeddings import load_cached_embeddings
from src.knn_graph import load_graph


def get_dataset():
    """Return (dataframe, subset_meta); stop the page if the dataset is missing."""
    if "subset" not in st.session_state:
        saved = load_saved_subset()
        if saved is not None:
            st.session_state["subset"], st.session_state["subset_meta"] = saved
    df = st.session_state.get("subset")
    if df is None:
        st.warning("Load the dataset first: open the Dataset page in the sidebar.")
        st.stop()
    return df, st.session_state["subset_meta"]


def get_embeddings(df, subset_meta):
    """Return (matrix, embedding_meta); stop the page if embeddings are missing."""
    held = st.session_state.get("embeddings_meta")
    if held is not None and held["dataset_fingerprint"] != subset_meta["fingerprint_sha256"]:
        for key in ("embeddings", "embeddings_meta", "embeddings_source"):
            st.session_state.pop(key, None)

    if "embeddings" not in st.session_state:
        matrix, meta, message = load_cached_embeddings(df)
        if matrix is None:
            st.warning(f"{message} Generate the embeddings on the Embeddings page first.")
            st.stop()
        st.session_state["embeddings"] = matrix
        st.session_state["embeddings_meta"] = meta
        st.session_state["embeddings_source"] = "Loaded from cache"
    return st.session_state["embeddings"], st.session_state["embeddings_meta"]


def get_graphs(source_key):
    """Dict k -> {graph, stats, source_key}; loads saved graphs from disk when valid."""
    store = st.session_state.setdefault("graphs", {})
    for k in config.K_VALUES:
        entry = store.get(k)
        if entry is not None and entry["source_key"] == source_key:
            continue
        loaded = load_graph(k, source_key)
        if loaded is None:
            store.pop(k, None)
        else:
            store[k] = {"graph": loaded[0], "stats": loaded[1], "source_key": source_key}
    return store