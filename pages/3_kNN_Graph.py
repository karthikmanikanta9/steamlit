import pandas as pd
import streamlit as st\n\nfrom src.ui import chart_template, init_page_ui

from src import config
from src.embeddings import most_similar
from src.knn_graph import (
    bfs_neighbourhood,
    build_knn_graph,
    embeddings_source_key,
    induced_subgraph,
    neighbour_lists,
    save_graph,
)
from src.state import get_dataset, get_embeddings, get_graphs
from src.visuals import network_figure, pca_layout

st.set_page_config(page_title="kNN Graph | MST Text Clustering", page_icon="🌳", layout="wide")\n\ninit_page_ui("kNN Graph")

st.title("kNN Graph Explorer")
st.caption("Every document is linked to the documents whose meaning is closest to it.")

df, subset_meta = get_dataset()
matrix, emb_meta = get_embeddings(df, subset_meta)
source_key = embeddings_source_key(emb_meta)
graphs = get_graphs(source_key)

with st.expander("How the graph is built"):
    st.markdown(
        "1. For each document, find its **k nearest neighbours** by cosine distance "
        "(exact search, nothing approximate).\n"
        "2. Each choice becomes a link. A link is **kept if either document chose the other** "
        "(*union symmetrization*).\n"
        "3. Duplicate links and self-links are removed, so each pair appears once.\n"
        "4. Each link gets **one weight**: the cosine distance between the two documents.\n\n"
        "**Smaller distance means more similar meaning.** "
        "Class labels are never used to build the graph."
    )


def build_and_store(k_value):
    graph, stats = build_knn_graph(matrix, k_value)
    save_graph(graph, stats, source_key)
    graphs[k_value] = {"graph": graph, "stats": stats, "source_key": source_key}
    st.session_state.get("mst_results", {}).pop(k_value, None)   # old MST results are now stale


# ---------- Build ----------
st.subheader("Build graphs")
k = st.radio("k = number of nearest neighbours chosen by each document",
             config.K_VALUES, horizontal=True, index=1)
left, right = st.columns(2)
try:
    if left.button(f"Build graph for k = {k}", type="primary"):
        with st.spinner(f"Building the k = {k} graph..."):
            build_and_store(k)
    if right.button("Build all four graphs (k = 5, 10, 15, 20)"):
        progress = st.progress(0.0)
        for number, k_value in enumerate(config.K_VALUES):
            progress.progress(number / len(config.K_VALUES), text=f"Building k = {k_value} ...")
            build_and_store(k_value)
        progress.progress(1.0, text="Done")
except Exception as error:
    st.error(f"Could not build the graph: {error}")

if not graphs:
    st.info("No graph built yet. Click a build button above.")
    st.stop()

# ---------- Overview of all built graphs ----------
st.subheader("Built graphs")
rows = []
for k_value in sorted(graphs):
    s = graphs[k_value]["stats"]
    rows.append({
        "k": k_value, "Vertices": s["vertices"], "Edges": s["edges"],
        "Density": round(s["density"], 5), "Average degree": round(s["average_degree"], 2),
        "Components": s["components"],
        "Status": "Connected" if s["connected"] else "Disconnected",
        "Build time (s)": s["build_seconds"],
    })
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("Build time covers the neighbour search and edge creation. Embedding generation is not included.")

if k not in graphs:
    st.info(f"The k = {k} graph is not built yet. Click the build button.")
    st.stop()

graph, stats = graphs[k]["graph"], graphs[k]["stats"]

# ---------- Selected graph ----------
st.subheader(f"Graph for k = {k}")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Vertices", f"{stats['vertices']:,}")
m2.metric("Edges", f"{stats['edges']:,}")
m3.metric("Density", f"{stats['density']:.5f}")
m4.metric("Components", stats["components"])
m5.metric("Status", "Connected" if stats["connected"] else "Disconnected")
m6.metric("Build time", f"{stats['build_seconds']:.2f} s")

if stats["connected"]:
    st.success(
        f"The graph is connected, so the algorithms will build a single spanning tree "
        f"with V - 1 = {stats['vertices'] - 1:,} edges."
    )
else:
    forest_edges = stats["vertices"] - stats["components"]
    st.warning(
        f"The graph has {stats['components']} separate components, so no single spanning tree exists. "
        f"The algorithms will build a minimum spanning forest with V - C = {forest_edges:,} edges. "
        "No edges are added to join the pieces."
    )
    sizes = pd.DataFrame({"Component": range(1, len(stats["component_sizes"]) + 1),
                          "Documents": stats["component_sizes"]})
    st.dataframe(sizes.head(10), hide_index=True)

with st.expander("Construction details"):
    st.write(
        f"- Neighbour choices made (n x k): **{stats['directed_selections']:,}**\n"
        f"- Self-links removed: **{stats['self_loops_removed']}**\n"
        f"- Pairs where both documents chose each other (merged into one edge): **{stats['mutual_pairs_merged']:,}**\n"
        f"- Final undirected edges: **{stats['edges']:,}**\n"
        f"- Degree (min / average / max): **{stats['min_degree']} / {stats['average_degree']:.2f} / {stats['max_degree']}**\n"
        f"- Edge weight = cosine distance (min / mean / max): "
        f"**{stats['weight_min']:.4f} / {stats['weight_mean']:.4f} / {stats['weight_max']:.4f}**\n"
        f"- Time: neighbour search {stats['search_seconds']:.3f} s, edge creation "
        f"{stats['edge_build_seconds']:.3f} s, statistics {stats['statistics_seconds']:.3f} s"
    )

# ---------- Neighbourhood explorer ----------
st.subheader("Neighbourhood explorer")
st.caption("Showing the whole graph would be unreadable, so pick a document and look around it.")
c1, c2, c3, c4 = st.columns(4)
class_choice = c1.selectbox("Class (to pick a document)", list(config.LABEL_NAMES.values()))
positions = df.index[df["label_name"] == class_choice].tolist()
number = c2.slider("Document number in class", 1, len(positions), 1)
hops = c3.radio("Steps around the document", [1, 2], horizontal=True)
max_nodes = c4.slider("Maximum documents shown", 10, 120, 40)
centre = positions[number - 1]

adjacency = neighbour_lists(graph)
nodes, depth = bfs_neighbourhood(adjacency, centre, hops, max_nodes)
sub = induced_subgraph(graph, nodes)

palette = {0: "#F59E0B", 1: "#22D3EE", 2: "#64748B"}
colors = [palette[depth[n]] for n in nodes]
hover = [f"{df.loc[n, 'doc_id']}<br>{df.loc[n, 'text'][:90]}" for n in nodes]
coords = pca_layout(matrix[nodes])
edges = [(a, b) for a, b in zip(sub.graph.u.tolist(), sub.graph.v.tolist())]
fig = network_figure(coords, edges, node_colors=colors, node_hover=hover)
st.plotly_chart(fig)
st.caption(
    f"{len(nodes)} documents and {sub.graph.m} links. Orange = selected document, cyan = 1 step away, "
    "grey = 2 steps away (colours show distance in the graph, not topic labels). "
    "Positions come from a 2-D projection of the embeddings, so nearby points tend to be similar."
)

st.markdown(f"**Selected document** (`{df.loc[centre, 'doc_id']}`):")
st.write(df.loc[centre, "text"])

own_top_k = set(most_similar(matrix, centre, top=k)[0].tolist())
table = []
for rank, (weight, neighbour) in enumerate(adjacency[centre][:30], start=1):
    table.append({
        "Rank": rank,
        "Cosine distance": round(weight, 4),
        "Why linked": "Among its own k nearest" if neighbour in own_top_k else "It chose this document",
        "Class": df.loc[neighbour, "label_name"],
        "Text": df.loc[neighbour, "text"][:120],
    })
st.markdown(f"**Linked documents** ({len(adjacency[centre])} in total, nearest first, up to 30 shown):")
st.dataframe(pd.DataFrame(table), hide_index=True)
st.caption("Classes are shown for illustration only; they are not used to build the graph.")