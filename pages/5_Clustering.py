"""Page 6 — MST-based Clustering.

Applies the size-constrained edge-cut heuristic to a chosen MST result:
  1. Sort MST edges heaviest → lightest.
  2. Remove an edge only if both resulting parts have ≥ min_size documents.
  3. Stop when target_k clusters are reached (or no eligible edge remains).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st\n\nfrom src.ui import chart_template, init_page_ui

from src.clustering import cluster_sizes, clusters_to_index_map, cut_mst_edges
from src.knn_graph import bfs_neighbourhood, induced_subgraph, neighbour_lists
from src import config
from src.mst import ALGORITHMS
from src.state import get_dataset, get_embeddings, get_graphs
from src.visuals import network_figure, pca_layout

st.set_page_config(
    page_title="Clustering | MST Text Clustering",
    page_icon="🌳",
    layout="wide",
)\n\ninit_page_ui("Clustering")

st.title("MST-Based Clustering")
st.caption(
    "Cut the heaviest MST edges (while keeping every part large enough) "
    "to form text clusters."
)

# ── Prerequisites ─────────────────────────────────────────────────────────────
df, subset_meta = get_dataset()
matrix, emb_meta = get_embeddings(df, subset_meta)

mst_store = st.session_state.get("mst_results", {})
if not mst_store:
    st.warning(
        "No MST results found. Run at least one algorithm on the "
        "**MST Algorithms** page first."
    )
    st.stop()

# ── Select MST result ──────────────────────────────────────────────────────────
st.subheader("Select an MST result")

available_ks = sorted(mst_store)
k = st.selectbox(
    "Graph (k value)",
    available_ks,
    format_func=lambda x: f"k = {x}",
)

available_algos = list(mst_store[k].keys())
if not available_algos:
    st.warning(f"No algorithm has been run on the k = {k} graph yet.")
    st.stop()

algo_name = st.radio(
    "Algorithm whose MST to cluster",
    available_algos,
    horizontal=True,
)

bench = mst_store[k][algo_name]
mst_result = bench.result
graph = st.session_state["graphs"][k]["graph"]

col1, col2, col3 = st.columns(3)
col1.metric("MST edges", f"{mst_result.num_edges:,}")
col2.metric("Total weight", f"{mst_result.total_weight:.5f}")
col3.metric("Vertices", f"{mst_result.num_vertices:,}")

# ── Edge-cut configuration ────────────────────────────────────────────────────
st.subheader("Edge-cut configuration")
st.markdown(
    "The MST itself does **not** decide the number of clusters. "
    "The **edge-cut rule** does: we remove the heaviest edges one at a time, "
    "but only when both resulting parts meet the minimum-size requirement."
)

cc1, cc2 = st.columns(2)
target_k = cc1.number_input(
    "Target cluster count",
    min_value=2,
    max_value=50,
    value=config.DEFAULT_N_CLUSTERS,
    step=1,
    help="Desired number of clusters. May not be reachable under the minimum-size constraint.",
)
min_size = cc2.number_input(
    "Minimum documents per cluster",
    min_value=1,
    max_value=500,
    value=config.DEFAULT_MIN_CLUSTER_SIZE,
    step=1,
    help="An edge is only removed if both resulting parts have at least this many documents.",
)

# ── Cache key for this clustering run ─────────────────────────────────────────
clust_key = (k, algo_name, int(target_k), int(min_size))
clust_store = st.session_state.setdefault("clustering_results", {})

# ── Run ───────────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 3])
if run_col.button("▶ Run edge-cut clustering", type="primary"):
    with st.spinner("Running size-constrained edge-cut…"):
        labels, cut_ids, actual_k, info = cut_mst_edges(
            graph, mst_result,
            target_k=int(target_k),
            min_size=int(min_size),
        )
    clust_store[clust_key] = {
        "labels": labels,
        "cut_edge_ids": cut_ids,
        "actual_k": actual_k,
        "info": info,
        "k": k,
        "algo": algo_name,
        "target_k": int(target_k),
        "min_size": int(min_size),
    }
    st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if clust_key not in clust_store:
    st.info("Configure the parameters above and click **Run edge-cut clustering**.")
    st.stop()

result = clust_store[clust_key]
labels    = result["labels"]
cut_ids   = result["cut_edge_ids"]
actual_k  = result["actual_k"]
info      = result["info"]

# Status banner
if info["reached_target"]:
    st.success(
        f"✅ Reached the target: **{actual_k} clusters** formed by cutting "
        f"{len(cut_ids)} edge(s)."
    )
else:
    st.warning(
        f"⚠️ Could only form **{actual_k} clusters** (target was {target_k}). "
        f"{info['skipped_count']} candidate edge(s) were skipped because one or "
        f"both resulting parts would have fewer than {min_size} documents."
    )

if info["initial_components"] > 1:
    st.info(
        f"ℹ️ The spanning forest started with **{info['initial_components']} "
        f"separate trees** (from the disconnected kNN graph). "
        "Each tree was treated as its own starting cluster."
    )

# ── Summary metrics ───────────────────────────────────────────────────────────
st.subheader("Cluster summary")
sizes = cluster_sizes(labels)
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Clusters formed", actual_k)
mc2.metric("Edges cut", len(cut_ids))
mc3.metric("Largest cluster", max(sizes.values()) if sizes else 0)
mc4.metric("Smallest cluster", min(sizes.values()) if sizes else 0)

# ── Cluster sizes bar chart ────────────────────────────────────────────────────
sizes_df = pd.DataFrame(
    {"Cluster": [f"C{k}" for k in sizes], "Documents": list(sizes.values())}
)
fig_bar = px.bar(
    sizes_df, x="Cluster", y="Documents",
    text="Documents", template=chart_template(),
    title="Documents per Cluster",
)
fig_bar.update_traces(marker_color="#22D3EE", textposition="outside")
fig_bar.update_layout(height=350, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_bar, use_container_width=True)

# ── Ground-truth topic distribution (evaluation only) ─────────────────────────
st.subheader("Ground-truth topic distribution per cluster")
st.caption(
    "These labels are shown **for evaluation only** and were never used "
    "to build embeddings, the kNN graph, or the MST."
)

idx_map = clusters_to_index_map(labels)
dist_rows = []
for cid in sorted(idx_map):
    idxs = idx_map[cid]
    sub_df = df.iloc[idxs]
    counts = sub_df["label_name"].value_counts()
    for topic, cnt in counts.items():
        dist_rows.append({
            "Cluster": f"C{cid}",
            "Topic": topic,
            "Documents": cnt,
        })

if dist_rows:
    dist_df = pd.DataFrame(dist_rows)
    fig_heat = px.bar(
        dist_df, x="Cluster", y="Documents", color="Topic",
        barmode="stack", template=chart_template(),
        title="Topic mix per cluster (evaluation only — labels not used in clustering)",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_heat.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Cluster sizes table ───────────────────────────────────────────────────────
st.subheader("Cluster details")
detail_rows = []
for cid in sorted(idx_map):
    idxs = idx_map[cid]
    sub_df = df.iloc[idxs]
    dominant = sub_df["label_name"].value_counts().idxmax() if len(sub_df) > 0 else "—"
    detail_rows.append({
        "Cluster": f"C{cid}",
        "Size": len(idxs),
        "Dominant topic (eval only)": dominant,
    })
st.dataframe(pd.DataFrame(detail_rows), hide_index=True)

# ── Sample articles per cluster ────────────────────────────────────────────────
st.subheader("Sample articles per cluster")
st.caption("5 randomly sampled articles from each cluster.")
view_cid = st.selectbox(
    "Inspect cluster",
    sorted(idx_map.keys()),
    format_func=lambda x: f"Cluster {x}  ({sizes[x]} documents)",
)
if view_cid in idx_map:
    sample_idxs = idx_map[view_cid][:5]
    for idx in sample_idxs:
        row = df.iloc[idx]
        st.markdown(
            f"**[{row['label_name']}]** `{row['doc_id']}`  \n{row['text'][:300]}…"
        )
        st.divider()

# ── Cut edges detail ──────────────────────────────────────────────────────────
if cut_ids:
    with st.expander(f"Cut edges ({len(cut_ids)} total)"):
        rows = []
        for rank, eid in enumerate(cut_ids, 1):
            rows.append({
                "Cut #": rank,
                "Edge ID": int(eid),
                "Weight (cosine dist)": round(float(graph.w[eid]), 6),
                "Vertex A": int(graph.u[eid]),
                "Vertex B": int(graph.v[eid]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

# ── Skipped edges ─────────────────────────────────────────────────────────────
if info["skipped_count"] > 0:
    with st.expander(f"Skipped edges ({info['skipped_count']} — size constraint not met)"):
        skip_rows = []
        for s in info["skipped_details"]:
            skip_rows.append({
                "Edge ID": s.get("edge_id", ""),
                "Weight": round(s.get("weight", 0.0), 6),
                "Part A size": s.get("size_a", ""),
                "Part B size": s.get("size_b", ""),
                "Reason": s.get("reason", ""),
            })
        st.dataframe(pd.DataFrame(skip_rows), hide_index=True)
        st.caption(
            "An edge is only removed when **both** resulting parts have "
            f"≥ {min_size} documents. Skipped edges violate this constraint."
        )

# ── Visual: MST subgraph coloured by cluster ──────────────────────────────────
st.subheader("Subgraph view — coloured by cluster")
st.caption(
    "A small neighbourhood is rendered. Node colours show cluster assignment, "
    "not ground-truth topic labels. Cut edges are shown in red."
)

PALETTE = [
    "#22D3EE", "#F59E0B", "#A78BFA", "#34D399", "#F87171",
    "#FB923C", "#60A5FA", "#E879F9", "#4ADE80", "#FACC15",
]

v_seed = st.number_input(
    "Centre document index", 0, graph.n - 1, 0, key="clust_seed"
)
v_size = st.slider("Subgraph size", 15, 80, 40, key="clust_size")

adj_full = neighbour_lists(graph)
nodes_list, _ = bfs_neighbourhood(adj_full, int(v_seed), v_size, v_size)
sub = induced_subgraph(graph, nodes_list)

node_colors = [
    PALETTE[int(labels[n]) % len(PALETTE)] for n in nodes_list
]

# Which edges in the sub are cut MST edges?
cut_set = set(cut_ids)
orig_eids = sub.original_edge_ids.tolist()
highlight_cut = np.array([bool(eid in cut_set) for eid in orig_eids])

hover = [
    f"Doc {int(n)} | Cluster {int(labels[n])}<br>{df.iloc[int(n)]['text'][:80]}"
    for n in nodes_list
]

coords = pca_layout(matrix[nodes_list])
sub_edges = list(zip(sub.graph.u.tolist(), sub.graph.v.tolist()))

fig_net = network_figure(
    coords, sub_edges,
    highlight=highlight_cut,
    node_colors=node_colors,
    node_hover=hover,
)
# Override highlight colour to red for cut edges
for trace in fig_net.data:
    if trace.line and trace.line.color == "#22D3EE":
        trace.line.color = "#EF4444"
        trace.line.width = 3.5

st.plotly_chart(fig_net, use_container_width=True)
st.caption(
    "Red lines = cut MST edges. Each node colour is a different cluster (not a topic label)."
)
