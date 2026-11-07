import pandas as pd
import plotly.express as px
import streamlit as st\n\nfrom src.ui import chart_template, init_page_ui

from src import config
from src.data_loader import create_and_save_subset, load_saved_subset, validate_subset

st.set_page_config(page_title="Dataset | MST Text Clustering", page_icon="🌳", layout="wide")\n\ninit_page_ui("Dataset")

st.title("Dataset")
st.caption("AG News: a balanced subset of news articles with four topics.")

# ---------- Configuration ----------
st.subheader("Configuration")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", config.DATASET_NAME)
c2.metric("Split used", config.DATASET_SPLIT)
c3.metric("Documents per class", config.SAMPLES_PER_CLASS)
c4.metric("Random seed", config.RANDOM_SEED)
st.caption(
    "Classes: " + ", ".join(config.LABEL_NAMES.values())
    + ". Labels are kept for evaluation only. They are never used to build "
    "embeddings, the kNN graph, or the MST."
)

# ---------- Load saved subset once per session ----------
if "subset" not in st.session_state:
    saved = load_saved_subset()
    if saved is not None:
        st.session_state["subset"], st.session_state["subset_meta"] = saved

# ---------- Action ----------
if st.button("Load AG News and create balanced subset", type="primary"):
    try:
        with st.spinner("Downloading AG News (first time only) and building the subset..."):
            df_new, meta_new = create_and_save_subset()
        st.session_state["subset"] = df_new
        st.session_state["subset_meta"] = meta_new
        st.success("Subset created and saved to data/processed/.")
    except Exception as error:
        st.error(f"Could not create the subset: {error}")

df = st.session_state.get("subset")
meta = st.session_state.get("subset_meta")

if df is None or meta is None:
    st.info("Dataset not loaded yet. Click the button above.")
    st.stop()

# ---------- Summary ----------
st.subheader("Selected subset")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total documents", f"{meta['total_documents']:,}")
m2.metric("Classes", len(meta["class_counts"]))
m3.metric("Removed records", meta["removed_empty_text"] + meta["removed_invalid_label"])
m4.metric("Fingerprint (first 12)", meta["fingerprint_sha256"][:12])
st.caption(
    f"Created {meta['created_utc']} (UTC). Same seed gives the same fingerprint, "
    "which shows the subset is reproducible."
)

# ---------- Class distribution ----------
st.subheader("Class distribution")
counts = (
    df["label_name"].value_counts()
    .reindex(list(config.LABEL_NAMES.values()))
    .rename_axis("Class")
    .reset_index(name="Documents")
)
left, right = st.columns([2, 1])
fig = px.bar(counts, x="Class", y="Documents", text="Documents", template=chart_template())
fig.update_traces(marker_color="#22D3EE", textposition="outside")
fig.update_layout(yaxis_range=[0, counts["Documents"].max() * 1.15])
left.plotly_chart(fig)
right.dataframe(counts, hide_index=True)

# ---------- Validation ----------
st.subheader("Validation")
for name, passed, detail in validate_subset(df, meta["samples_per_class"]):
    icon = "✅" if passed else "❌"
    st.write(f"{icon} {name}" + (f" ({detail})" if detail else ""))

# ---------- Preprocessing ----------
st.subheader("Preprocessing summary")
for step in meta["preprocessing_steps"]:
    st.markdown(f"- {step}")
st.caption(
    f"Raw rows: {meta['raw_rows']:,}. Removed for empty text: {meta['removed_empty_text']}. "
    f"Removed for invalid label: {meta['removed_invalid_label']}."
)

# ---------- Samples ----------
st.subheader("Sample documents")
choice = st.selectbox("Filter by class", ["All"] + list(config.LABEL_NAMES.values()))
view = df if choice == "All" else df[df["label_name"] == choice]
st.dataframe(view[["doc_id", "label_name", "text"]].head(10), hide_index=True)
st.caption(
    "In AG News the article title and description are stored together in one text field."
)