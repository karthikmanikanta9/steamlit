"""MST-Based Clustering of Text Embeddings.

Academic Dashboard & Experimental Platform (DAA Project).
"""
from __future__ import annotations

import streamlit as st

from src import config
from src.data_loader import load_saved_subset
from src.embeddings import read_embeddings_meta
from src.experiment_store import load_experiments
from src.knn_graph import embeddings_source_key, saved_graph_ks

st.set_page_config(
    page_title="MST Text Clustering — DAA Dashboard",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar Branding ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌳 MST Text Clustering")
    st.caption("Design & Analysis of Algorithms Project")
    st.markdown("---")
    st.markdown("**Navigation Guide**")
    st.markdown("""
    1. **Dataset**: Balance 3,000 AG News articles
    2. **Embeddings**: MiniLM 384-D vectors
    3. **kNN Graph**: Exact cosine similarity graph
    4. **MST Algorithms**: Prim vs. Kruskal vs. Borůvka
    5. **Clustering**: Size-constrained edge-cutting
    6. **Baseline Comparison**: K-means & HDBSCAN
    7. **Experimental Results**: Metrics, history & Exp C
    8. **Methodology**: Detailed theoretical foundations
    """)
    st.markdown("---")
    st.caption("SRM University-AP • DAA Coursework")

# ── Header & Academic Context ─────────────────────────────────────────────────
st.title("MST-Based Clustering of Text Embeddings")
st.caption("An Empirical Study of Prim's, Kruskal's, and Borůvka's Algorithms on Sparse kNN Graphs")

st.markdown("""
> **Research Question:**  
> *How do Prim's, Kruskal's, and Borůvka's algorithms compare when constructing Minimum Spanning Trees 
> on kNN graphs of text embeddings, and how do the resulting clusters compare with K-means and HDBSCAN 
> in efficiency and clustering quality?*
""")

# ── Pipeline Status Checks ────────────────────────────────────────────────────
saved = load_saved_subset()
has_dataset = saved is not None
dataset_status = f"{len(saved[0]):,} docs" if has_dataset else "Not loaded"

embedding_meta = read_embeddings_meta()
embeddings_ready = bool(
    has_dataset and embedding_meta
    and embedding_meta.get("dataset_fingerprint") == saved[1]["fingerprint_sha256"]
)
embedding_status = (
    f"{embedding_meta['num_documents']:,} × {embedding_meta['dimension']}D"
    if embeddings_ready else "Not generated"
)

built_ks = saved_graph_ks(embeddings_source_key(embedding_meta)) if embeddings_ready else []
graph_status = f"k = {', '.join(map(str, built_ks))}" if built_ks else "Not built"

mst_runs = sum(len(runs) for runs in st.session_state.get("mst_results", {}).values())
mst_status = f"{mst_runs} run(s)" if mst_runs else "Not run"

clust_runs = len(st.session_state.get("clustering_results", {}))
clust_status = f"{clust_runs} run(s)" if clust_runs else "Not run"

baselines_done = bool(
    st.session_state.get("km_labels") is not None
    or st.session_state.get("hd_labels") is not None
)
baseline_status = "Computed" if baselines_done else "Not run"

saved_experiments = load_experiments()

# ── Pipeline Progression Cards ────────────────────────────────────────────────
st.subheader("Pipeline Progress Tracker")

col_p1, col_p2, col_p3, col_p4, col_p5 = st.columns(5)

with col_p1:
    icon1 = "✅" if has_dataset else "⏳"
    st.metric(f"{icon1} 1. Dataset", dataset_status)
    st.caption("AG News 4 classes")

with col_p2:
    icon2 = "✅" if embeddings_ready else "⏳"
    st.metric(f"{icon2} 2. Embeddings", embedding_status)
    st.caption("all-MiniLM-L6-v2")

with col_p3:
    icon3 = "✅" if built_ks else "⏳"
    st.metric(f"{icon3} 3. kNN Graph", graph_status)
    st.caption("Union symmetrized")

with col_p4:
    icon4 = "✅" if mst_runs > 0 else "⏳"
    st.metric(f"{icon4} 4. MST / MSF", mst_status)
    st.caption("Prim, Kruskal, Borůvka")

with col_p5:
    icon5 = "✅" if clust_runs > 0 and baselines_done else "⏳"
    st.metric(f"{icon5} 5. Evaluation", f"{len(saved_experiments)} logs")
    st.caption("ARI, NMI, Silhouette")

st.divider()

# ── Actionable Workflow Guidance ──────────────────────────────────────────────
st.subheader("Current Step Guidance")
if not has_dataset:
    st.info("👉 **Step 1:** Open **1_Dataset** in the sidebar to download and balance 3,000 articles from AG News.")
elif not embeddings_ready:
    st.info("👉 **Step 2:** Open **2_Embeddings** to encode texts into 384-dimensional unit-norm vectors.")
elif not built_ks:
    st.info("👉 **Step 3:** Open **3_kNN_Graph** to construct sparse cosine similarity graphs for k ∈ {5, 10, 15, 20}.")
elif not mst_runs:
    st.info("👉 **Step 4:** Open **4_MST_Algorithms** to benchmark and validate Prim, Kruskal, and Borůvka from scratch.")
elif not clust_runs:
    st.info("👉 **Step 5:** Open **5_Clustering** to apply the size-constrained edge-cutting heuristic.")
elif not baselines_done:
    st.info("👉 **Step 6:** Open **6_Baseline_Comparison** to run K-means and HDBSCAN baselines and save results.")
else:
    st.success("🎉 **Full pipeline active!** Head to **7_Experimental_Results** to analyze performance curves and Experiment C.")

# ── Architecture & Methodology Highlights ─────────────────────────────────────
st.divider()
st.subheader("Key Architectural Highlights")

c_arch1, c_arch2, c_arch3 = st.columns(3)

with c_arch1:
    st.markdown("#### 🌲 MST Algorithms")
    st.markdown("""
    - **Prim's Algorithm**: Binary min-heap, priority queue over cut edges.
    - **Kruskal's Algorithm**: Disjoint Set Union (DSU) with path compression & union by size.
    - **Borůvka's Algorithm**: Multi-phase component contraction with $O(\\log V)$ phases.
    - **Deterministic Tie-Breaking**: Strict total order via `(weight, edge_id)`.
    """)

with c_arch2:
    st.markdown("#### ✂️ Size-Constrained Edge-Cut")
    st.markdown("""
    - Sorts candidate MST edges from highest weight to lowest weight.
    - Cuts an edge **only if** both resulting subtrees satisfy $N_{\\text{size}} \\ge \\text{min\\_size}$.
    - Resolves arbitrary-shape single-linkage clusters while preventing singleton outlier splits.
    - Disconnected spanning forests naturally form independent starting clusters.
    """)

with c_arch3:
    st.markdown("#### ⚖️ Fair Baseline Comparison")
    st.markdown("""
    - **K-means**: Operates directly on embedding space (default $k=4$, cosine geometry).
    - **HDBSCAN**: Density-based clustering with outlier/noise point extraction.
    - **No Label Leaking**: Ground truth topic labels are retained strictly for ARI and NMI computation.
    - **Experiment C**: Empirical comparison of Complete $O(n^2)$ vs. Sparse $O(nk)$ graphs.
    """)

st.divider()
st.caption("Antigravity DAA Platform • Clean Python 3 • Fully Modular • Local Execution")
