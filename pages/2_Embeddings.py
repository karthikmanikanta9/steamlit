import pandas as pd
import streamlit as st

from src import config
from src.data_loader import load_saved_subset
from src.embeddings import (
    cosine_distance,
    create_and_save_embeddings,
    load_cached_embeddings,
    most_similar,
    validate_embeddings,
)

st.set_page_config(page_title="Embeddings | MST Text Clustering", page_icon="🌳", layout="wide")

st.title("Embeddings")
st.caption("Each article is turned into a list of numbers (a vector) that captures its meaning.")

# ---------- Dataset needed first ----------
if "subset" not in st.session_state:
    saved = load_saved_subset()
    if saved is not None:
        st.session_state["subset"], st.session_state["subset_meta"] = saved

df = st.session_state.get("subset")
if df is None:
    st.warning("Load the dataset first: open the Dataset page in the sidebar.")
    st.stop()

current_fingerprint = st.session_state["subset_meta"]["fingerprint_sha256"]

# Drop embeddings in memory if the dataset subset has changed.
held = st.session_state.get("embeddings_meta")
if held is not None and held["dataset_fingerprint"] != current_fingerprint:
    for key in ("embeddings", "embeddings_meta", "embeddings_source"):
        st.session_state.pop(key, None)

# ---------- Configuration ----------
st.subheader("Configuration")
st.markdown(f"**Model:** `{config.EMBEDDING_MODEL}`")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Documents to embed", f"{len(df):,}")
c2.metric("Device", config.EMBEDDING_DEVICE.upper())
c3.metric("Normalisation", "Unit length")
c4.metric("Batch size", config.EMBEDDING_BATCH_SIZE)

# ---------- Load from cache automatically (cheap), never generate automatically ----------
cache_message = ""
if "embeddings" not in st.session_state:
    matrix_cached, meta_cached, cache_message = load_cached_embeddings(df)
    if matrix_cached is not None:
        st.session_state["embeddings"] = matrix_cached
        st.session_state["embeddings_meta"] = meta_cached
        st.session_state["embeddings_source"] = "Loaded from cache"

# ---------- Action ----------
already = "embeddings" in st.session_state
button_label = "Regenerate embeddings (overwrites the cache)" if already else "Generate embeddings"
if st.button(button_label, type="secondary" if already else "primary"):
    progress = st.progress(0.0, text="Loading the model (the first run downloads about 90 MB)...")

    def on_progress(done, total):
        progress.progress(done / total, text=f"Embedding documents: {done:,} / {total:,}")

    try:
        matrix_new, meta_new = create_and_save_embeddings(df, on_progress)
        st.session_state["embeddings"] = matrix_new
        st.session_state["embeddings_meta"] = meta_new
        st.session_state["embeddings_source"] = "Generated just now"
        progress.progress(1.0, text="Done")
        st.success("Embeddings generated and saved to data/embeddings/.")
    except Exception as error:
        progress.empty()
        st.error(f"Could not generate embeddings: {error}")

if "embeddings" not in st.session_state:
    st.info(cache_message or "No embeddings yet. Click the button above.")
    st.stop()

matrix = st.session_state["embeddings"]
meta = st.session_state["embeddings_meta"]

# ---------- Summary ----------
st.subheader("Result")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Documents", f"{matrix.shape[0]:,}")
m2.metric("Dimensions", matrix.shape[1])
m3.metric("Generation time", f"{meta['encode_seconds']:.1f} s")
m4.metric("Cache status", st.session_state["embeddings_source"])
size_mb = config.EMBEDDINGS_NPZ.stat().st_size / 1_000_000 if config.EMBEDDINGS_NPZ.exists() else 0
st.caption(
    f"Created {meta['created_utc']} (UTC). Model load time: {meta['model_load_seconds']:.1f} s "
    f"(not counted in generation time). Saved file: {size_mb:.1f} MB. "
    "The generation time shown is from the run that created the saved embeddings."
)
if meta.get("max_seq_length"):
    st.caption(f"The model reads at most {meta['max_seq_length']} tokens per document; longer text is cut off.")

# ---------- Checks ----------
st.subheader("Validation")
for name, passed, detail in validate_embeddings(matrix, len(df)):
    st.write(f"{'✅' if passed else '❌'} {name}" + (f" ({detail})" if detail else ""))

# ---------- Cosine distance ----------
st.subheader("Cosine distance")
st.latex(r"d(u,v) = 1 - \cos(u,v) = 1 - \frac{u \cdot v}{\|u\|\,\|v\|}")
st.caption(
    "Smaller distance means more similar meaning. Our vectors have length 1, so "
    "the distance is simply 1 minus the dot product."
)

# ---------- Semantic similarity example ----------
st.subheader("Semantic similarity example")
st.caption("Pick a document and see which other documents have the closest meaning.")
class_choice = st.selectbox("Class", list(config.LABEL_NAMES.values()))
positions = df.index[df["label_name"] == class_choice].tolist()
number = st.slider("Document number within the class", 1, len(positions), 1)
query = positions[number - 1]

st.markdown(f"**Selected document** (`{df.loc[query, 'doc_id']}`, {df.loc[query, 'label_name']}):")
st.write(df.loc[query, "text"])

order, similarities = most_similar(matrix, query, top=5)
result = pd.DataFrame({
    "Rank": range(1, len(order) + 1),
    "Cosine similarity": similarities.round(4),
    "Cosine distance": cosine_distance(similarities).round(4),
    "Class": df.iloc[order]["label_name"].values,
    "Text": df.iloc[order]["text"].str.slice(0, 140).values,
})
st.dataframe(result, hide_index=True)
same = int((df.iloc[order]["label_name"].values == df.loc[query, "label_name"]).sum())
st.caption(
    f"{same} of the 5 closest documents are from the same class. "
    "Class labels are shown here only for illustration; they are never used to build embeddings."
)