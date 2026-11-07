import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import chart_template, init_page_ui

from src import config
from src.benchmark import run_and_validate
from src.knn_graph import (
    bfs_neighbourhood,
    embeddings_source_key,
    induced_subgraph,
    neighbour_lists,
)
from src.mst import ALGORITHMS
from src.mst_validation import all_passed, compare_results, scipy_reference_weight, validate_mst
from src.state import get_dataset, get_embeddings, get_graphs
from src.visuals import network_figure, pca_layout

st.set_page_config(page_title="MST Algorithms | MST Text Clustering", page_icon="🌳", layout="wide")

init_page_ui("MST Algorithms")

st.title("MST Algorithms")
st.caption("Prim, Kruskal and Borůvka, written from scratch and run on the same graph.")

df, subset_meta = get_dataset()
matrix, emb_meta = get_embeddings(df, subset_meta)
source_key = embeddings_source_key(emb_meta)
graphs = get_graphs(source_key)

if not graphs:
    st.warning("Build a kNN graph first: open the kNN Graph page in the sidebar.")
    st.stop()

store = st.session_state.setdefault("mst_results", {})
if st.session_state.get("mst_results_source") != source_key:
    store.clear()
    st.session_state["mst_results_source"] = source_key

# ---------- Choose graph ----------
k = st.selectbox(
    "Graph", sorted(graphs),
    format_func=lambda x: f"k = {x}  ({graphs[x]['stats']['vertices']:,} vertices, {graphs[x]['stats']['edges']:,} edges)",
)
graph, stats = graphs[k]["graph"], graphs[k]["stats"]
results = store.setdefault(k, {})

if stats["connected"]:
    st.success(f"Connected graph: each algorithm should select V - 1 = {graph.n - 1:,} edges (one tree).")
else:
    st.warning(
        f"This graph has {stats['components']} components, so the result is a minimum spanning "
        f"forest with V - C = {graph.n - stats['components']:,} edges. No edges are added between components."
    )

with st.expander("How each algorithm works"):
    st.markdown(
        "- **Prim**: start at one document, then repeatedly take the cheapest edge that reaches a "
        "document outside the tree (binary heap). In a forest it restarts in each new component.\n"
        "- **Kruskal**: sort all edges by weight and add an edge only if it joins two different "
        "groups (Union-Find with path compression and union by size).\n"
        "- **Borůvka**: every group picks its cheapest outgoing edge, all those edges are added, "
        "groups merge, and this repeats. The number of groups at least halves each phase.\n\n"
        "**Ties:** edges are compared by (weight, edge ID), so there is always one clear cheapest edge."
    )

# ---------- Run controls ----------
st.subheader("Run on the full graph")
c1, c2 = st.columns(2)
repeats = c1.slider("Timed runs per algorithm (median is reported)", 1, 10, config.TIMING_REPEATS)
start_vertex = c2.number_input("Prim start vertex", 0, graph.n - 1, 0)

b1, b2, b3, b4 = st.columns(4)
to_run = []
if b1.button("Run Prim"):
    to_run = ["Prim"]
if b2.button("Run Kruskal"):
    to_run = ["Kruskal"]
if b3.button("Run Borůvka"):
    to_run = ["Borůvka"]
if b4.button("Run all three", type="primary"):
    to_run = list(ALGORITHMS)

if to_run:
    try:
        entry = graphs[k]
        if "reference_weight" not in entry:
            entry["reference_weight"] = scipy_reference_weight(graph)
        progress = st.progress(0.0)
        for number, name in enumerate(to_run):
            progress.progress(number / len(to_run), text=f"Running {name}: {repeats} timed runs + 1 memory run ...")
            kwargs = {"start": int(start_vertex)} if name == "Prim" else {}
            results[name] = run_and_validate(name, ALGORITHMS[name], graph, repeats,
                                             entry["reference_weight"], **kwargs)
        progress.progress(1.0, text="Finished")
    except Exception as error:
        st.error(f"Run failed: {error}")

# ---------- Results ----------
if not results:
    st.info("No results yet for this graph. Click a Run button.")
else:
    st.subheader("Results")
    rows = []
    for name in ALGORITHMS:
        if name in results:
            bench = results[name]
            rows.append({
                "Algorithm": name,
                "Edges selected": bench.result.num_edges,
                "Total weight": round(bench.result.total_weight, 8),
                "Median time (ms)": round(bench.median_seconds * 1000, 2),
                "Fastest run (ms)": round(bench.min_seconds * 1000, 2),
                "Peak memory (MB)": round(bench.peak_memory_bytes / 1e6, 2),
                "Timed runs": bench.repeats,
                "Phases (Borůvka)": str(bench.result.phases) if bench.result.phases is not None else "-",
                "Validation": "✅ passed" if bench.valid else "❌ failed",
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    if len(results) >= 2:
        report = compare_results({n: b.result for n, b in results.items()})
        st.write(f"{'✅' if report['weights_agree'] else '❌'} All algorithms agree on the total weight "
                 f"(largest difference {report['max_weight_difference']:.2e}).")
        st.write(f"{'✅' if report['identical_edge_sets'] else 'ℹ️'} "
                 f"{'They selected exactly the same edges.' if report['identical_edge_sets'] else 'They selected different edge sets of equal total weight.'}")
    else:
        st.caption("Run at least two algorithms to compare their total weights.")

    with st.expander("Validation details"):
        for name, bench in results.items():
            st.markdown(f"**{name}**")
            for check, passed, detail in bench.checks:
                st.write(f"{'✅' if passed else '❌'} {check}" + (f" ({detail})" if detail else ""))

    with st.expander("How time and memory are measured"):
        st.markdown(
            "- **Time:** each algorithm runs the chosen number of times on the same read-only graph; "
            "the median is shown. Each run includes the algorithm's own setup (sorting edges, building "
            "adjacency lists, and so on) because that work belongs to the algorithm.\n"
            "- **Memory:** one extra run measured with `tracemalloc`; this is the peak memory the "
            "algorithm itself allocated while running. The input graph and the embeddings are not counted.\n"
            "- The kNN graph construction and the embedding generation are **not** part of these times."
        )

    first = next(iter(results.values())).result
    weights = graph.w[first.edge_ids]
    hist = px.histogram(x=weights, nbins=50, template=chart_template(),
                        labels={"x": "Edge weight (cosine distance)", "y": "Number of MST edges"})
    hist.update_traces(marker_color="#22D3EE")
    hist.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    st.markdown("**Weights of the selected edges**")
    st.plotly_chart(hist)
    st.caption("The few heaviest edges at the right are the weakest links in the tree. They are the "
               "candidates that the clustering step cuts.")

# ---------- Step-by-step ----------
st.subheader("Step-by-step view (small subgraph)")
st.caption("Animating 3,000 documents is not readable, so this view runs the chosen algorithm on a small "
           "connected piece of the graph around one document.")
a1, a2, a3 = st.columns(3)
algo = a1.radio("Algorithm", list(ALGORITHMS), horizontal=True, key="viz_algo")
size = a2.slider("Subgraph size (documents)", 10, 60, 30)
seed_vertex = int(a3.number_input("Start document index", 0, graph.n - 1, 0, key="viz_seed"))

adjacency = neighbour_lists(graph)
nodes, _ = bfs_neighbourhood(adjacency, seed_vertex, size, size)
sub = induced_subgraph(graph, nodes)
sub_result = ALGORITHMS[algo](sub.graph)       # Prim starts at local vertex 0 = the start document
sub_checks = validate_mst(sub.graph, sub_result, scipy_reference_weight(sub.graph))
st.write(f"{'✅' if all_passed(sub_checks) else '❌'} {algo} on this subgraph: {sub_result.num_edges} edges, "
         f"total weight {sub_result.total_weight:.5f}, validated against an independent check.")

if sub_result.num_edges == 0:
    st.info("This subgraph has no edges. Try another start document.")
    st.stop()

step = st.slider("Edges accepted so far", 0, sub_result.num_edges, 0)
accepted = sub_result.edge_ids[:step]
mask = np.zeros(sub.graph.m, dtype=bool)
mask[accepted] = True
touched = set(sub.graph.u[accepted].tolist()) | set(sub.graph.v[accepted].tolist())
colors = ["#22D3EE" if i in touched else "#64748B" for i in range(len(nodes))]
colors[0] = "#F59E0B"                                   # the start document
hover = [f"Document {int(n)}<br>{df.loc[int(n), 'text'][:90]}" for n in nodes]
fig = network_figure(pca_layout(matrix[nodes]),
                     list(zip(sub.graph.u.tolist(), sub.graph.v.tolist())),
                     highlight=mask, node_colors=colors, node_hover=hover)
st.plotly_chart(fig)
st.caption("Bright lines = edges accepted so far, faint lines = edges not (yet) chosen. "
           "Orange = start document, cyan = already connected, grey = not yet reached.")

if step:
    log = []
    for number, edge_id in enumerate(accepted, start=1):
        row = {"Step": number,
               "Document A": int(nodes[sub.graph.u[edge_id]]),
               "Document B": int(nodes[sub.graph.v[edge_id]]),
               "Weight": round(float(sub.graph.w[edge_id]), 5)}
        if sub_result.phase_ends:
            row["Phase"] = next(p + 1 for p, end in enumerate(sub_result.phase_ends) if number <= end)
        log.append(row)
    st.dataframe(pd.DataFrame(log), hide_index=True)