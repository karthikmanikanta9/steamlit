"""Page 7 — Baseline Comparison.

K-means and HDBSCAN operate DIRECTLY on the same embedding matrix.
They do NOT use the kNN graph or MST.

Baseline path : embeddings → K-means / HDBSCAN → labels
MST path      : embeddings → kNN graph → MST → edge-cut → labels

Evaluation metrics (ARI, NMI, Silhouette) compare each method's cluster
assignments against the AG News ground-truth labels.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.ui import chart_template, init_page_ui

from src import config
from src.baselines import run_hdbscan, run_kmeans
from src.evaluation import evaluate_all
from src.experiment_store import save_experiment
from src.state import get_dataset, get_embeddings

st.set_page_config(
    page_title="Baseline Comparison | MST Text Clustering",
    page_icon="🌳",
    layout="wide",
)

init_page_ui("Baseline Comparison")

st.title("Baseline Comparison")
st.caption(
    "K-means and HDBSCAN cluster the **same embeddings** directly. "
    "No kNN graph or MST is involved in either baseline."
)

# ── Prerequisites ─────────────────────────────────────────────────────────────
df, subset_meta = get_dataset()
matrix, emb_meta = get_embeddings(df, subset_meta)
true_labels = df["label"].to_numpy(dtype=np.int64)

n_docs, n_dim = matrix.shape
st.info(
    f"Using **{n_docs:,} document embeddings** ({n_dim} dimensions) "
    f"from model `{emb_meta['model']}`.  "
    "Ground-truth labels are used **only for evaluation** (ARI / NMI), not for clustering."
)

# ── MST clustering result (optional comparison) ───────────────────────────────
clust_store = st.session_state.get("clustering_results", {})
mst_clust = None
mst_clust_key = None
if clust_store:
    keys = list(clust_store.keys())
    # pick the most recently run result
    mst_clust = clust_store[keys[-1]]
    mst_clust_key = keys[-1]

# ── K-MEANS ───────────────────────────────────────────────────────────────────
st.subheader("K-means")
km_col1, km_col2, km_col3 = st.columns(3)
km_k       = km_col1.number_input("Number of clusters", 2, 50, config.DEFAULT_N_CLUSTERS, key="km_k")
km_seed    = km_col2.number_input("Random seed", 0, 9999, config.KMEANS_SEED, key="km_seed")
km_n_init  = km_col3.number_input("n_init (restarts)", 1, 50, config.KMEANS_N_INIT, key="km_ninit")

if st.button("▶ Run K-means", type="primary", key="btn_kmeans"):
    with st.spinner("Running K-means…"):
        km_labels, km_info = run_kmeans(
            matrix,
            n_clusters=int(km_k),
            seed=int(km_seed),
            n_init=int(km_n_init),
        )
        km_metrics = evaluate_all(true_labels, km_labels, matrix)
    st.session_state["km_labels"]  = km_labels
    st.session_state["km_info"]    = km_info
    st.session_state["km_metrics"] = km_metrics
    st.rerun()

if "km_labels" in st.session_state:
    km_labels  = st.session_state["km_labels"]
    km_info    = st.session_state["km_info"]
    km_metrics = st.session_state["km_metrics"]

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Clusters", km_info["n_clusters_actual"])
    mc2.metric("ARI",        f"{km_metrics['ari']:.4f}" if km_metrics['ari'] is not None else "—")
    mc3.metric("NMI",        f"{km_metrics['nmi']:.4f}" if km_metrics['nmi'] is not None else "—")
    mc4.metric("Silhouette", f"{km_metrics['silhouette']:.4f}" if km_metrics['silhouette'] is not None else "—")
    mc5.metric("Time",       f"{km_info['runtime_seconds']*1000:.1f} ms")

    # Cluster sizes
    km_sizes_df = pd.DataFrame({
        "Cluster": [f"C{k}" for k in km_info["cluster_sizes"]],
        "Documents": list(km_info["cluster_sizes"].values()),
    })
    fig_km = px.bar(
        km_sizes_df, x="Cluster", y="Documents",
        text="Documents", template=chart_template(),
        title="K-means — Cluster sizes",
    )
    fig_km.update_traces(marker_color="#22D3EE", textposition="outside")
    fig_km.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_km, use_container_width=True)

else:
    st.info("K-means has not been run yet.")

st.divider()

# ── HDBSCAN ───────────────────────────────────────────────────────────────────
st.subheader("HDBSCAN")
st.caption(
    "HDBSCAN may internally use an MST as part of its own algorithm. "
    "It is still an independent baseline here — it does **not** use "
    "the MST built in Phase 5."
)
hd_col1, hd_col2 = st.columns(2)
hd_mcs = hd_col1.number_input(
    "min_cluster_size",
    min_value=2, max_value=500, value=config.HDBSCAN_MIN_CLUSTER_SIZE, key="hd_mcs",
    help="Minimum number of documents to form a cluster.",
)
hd_ms = hd_col2.number_input(
    "min_samples",
    min_value=1, max_value=200, value=config.HDBSCAN_MIN_SAMPLES, key="hd_ms",
    help="Controls cluster conservativeness. Set equal to min_cluster_size for default behaviour.",
)

if st.button("▶ Run HDBSCAN", type="primary", key="btn_hdbscan"):
    with st.spinner("Running HDBSCAN…"):
        try:
            hd_labels, hd_info = run_hdbscan(
                matrix,
                min_cluster_size=int(hd_mcs),
                min_samples=int(hd_ms),
            )
            # Evaluate only on non-noise points
            hd_metrics = evaluate_all(true_labels, hd_labels, matrix)
            st.session_state["hd_labels"]  = hd_labels
            st.session_state["hd_info"]    = hd_info
            st.session_state["hd_metrics"] = hd_metrics
        except ImportError as exc:
            st.error(str(exc))
    st.rerun()

if "hd_labels" in st.session_state:
    hd_labels  = st.session_state["hd_labels"]
    hd_info    = st.session_state["hd_info"]
    hd_metrics = st.session_state["hd_metrics"]

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.metric("Clusters",   hd_info["n_clusters"])
    mc2.metric("Noise pts",  hd_info["noise_points"])
    mc3.metric("ARI",        f"{hd_metrics['ari']:.4f}" if hd_metrics['ari'] is not None else "—")
    mc4.metric("NMI",        f"{hd_metrics['nmi']:.4f}" if hd_metrics['nmi'] is not None else "—")
    mc5.metric("Silhouette", f"{hd_metrics['silhouette']:.4f}" if hd_metrics['silhouette'] is not None else "—")
    mc6.metric("Time",       f"{hd_info['runtime_seconds']*1000:.1f} ms")

    if hd_info["noise_points"] > 0:
        st.warning(
            f"**{hd_info['noise_points']} noise points** ({hd_info['noise_fraction']*100:.1f}% "
            "of documents) were labelled as outliers by HDBSCAN and excluded from ARI/NMI/silhouette."
        )

    if hd_info["cluster_sizes"]:
        hd_sizes_df = pd.DataFrame({
            "Cluster": [f"C{k}" for k in hd_info["cluster_sizes"]],
            "Documents": list(hd_info["cluster_sizes"].values()),
        })
        fig_hd = px.bar(
            hd_sizes_df, x="Cluster", y="Documents",
            text="Documents", template=chart_template(),
            title="HDBSCAN — Cluster sizes (noise excluded)",
        )
        fig_hd.update_traces(marker_color="#A78BFA", textposition="outside")
        fig_hd.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_hd, use_container_width=True)
    else:
        st.warning("HDBSCAN found no clusters. Try a smaller `min_cluster_size`.")
else:
    st.info("HDBSCAN has not been run yet.")

st.divider()

# ── Comparison table ──────────────────────────────────────────────────────────
st.subheader("Comparison")

rows = []

# MST-based clustering
if mst_clust is not None:
    mst_labels = mst_clust["labels"]
    mst_metrics = evaluate_all(true_labels, mst_labels, matrix)
    rows.append({
        "Method": f"MST ({mst_clust['algo']}, k={mst_clust['k']})",
        "# Clusters": mst_clust["actual_k"],
        "ARI":        mst_metrics["ari"],
        "NMI":        mst_metrics["nmi"],
        "Silhouette": mst_metrics["silhouette"],
        "Noise pts":  0,
        "Note": "Edge-cut heuristic on MST",
    })

if "km_labels" in st.session_state:
    rows.append({
        "Method": "K-means",
        "# Clusters": st.session_state["km_info"]["n_clusters_actual"],
        "ARI":        st.session_state["km_metrics"]["ari"],
        "NMI":        st.session_state["km_metrics"]["nmi"],
        "Silhouette": st.session_state["km_metrics"]["silhouette"],
        "Noise pts":  0,
        "Note": f"seed={st.session_state['km_info']['seed']}",
    })

if "hd_labels" in st.session_state:
    rows.append({
        "Method": "HDBSCAN",
        "# Clusters": st.session_state["hd_info"]["n_clusters"],
        "ARI":        st.session_state["hd_metrics"]["ari"],
        "NMI":        st.session_state["hd_metrics"]["nmi"],
        "Silhouette": st.session_state["hd_metrics"]["silhouette"],
        "Noise pts":  st.session_state["hd_info"]["noise_points"],
        "Note": f"mcs={st.session_state['hd_info']['min_cluster_size']}",
    })

if rows:
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, hide_index=True, use_container_width=True)

    if len(rows) >= 2:
        fig_cmp = px.bar(
            cmp_df.melt(
                id_vars=["Method"],
                value_vars=["ARI", "NMI", "Silhouette"],
                var_name="Metric", value_name="Score",
            ),
            x="Metric", y="Score", color="Method",
            barmode="group", template=chart_template(),
            title="Clustering quality comparison",
            color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
        )
        fig_cmp.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig_cmp, use_container_width=True)
else:
    st.info("Run at least one method to see the comparison.")

# ── Save experiment ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Save experiment record")
if rows:
    if st.button("💾 Save this comparison to experiment log"):
        mst_bench = None
        if mst_clust_key and mst_clust:
            k_val, algo_val = mst_clust["k"], mst_clust["algo"]
            mst_bench = st.session_state["mst_results"].get(k_val, {}).get(algo_val)

        record: dict = {
            "dataset_size": int(len(df)),
            "k": mst_clust["k"] if mst_clust else None,
            "algorithm": mst_clust["algo"] if mst_clust else None,
            "mst_runtime_seconds": mst_bench.median_seconds if mst_bench else None,
            "mst_peak_memory_bytes": mst_bench.peak_memory_bytes if mst_bench else None,
            "mst_total_weight": float(mst_bench.result.total_weight) if mst_bench else None,
            "n_clusters": mst_clust["actual_k"] if mst_clust else None,
            "target_k": mst_clust["target_k"] if mst_clust else None,
            "min_size": mst_clust["min_size"] if mst_clust else None,
        }
        if mst_clust:
            m = evaluate_all(true_labels, mst_clust["labels"], matrix)
            record.update({"ari": m["ari"], "nmi": m["nmi"], "silhouette": m["silhouette"]})
        if "km_info" in st.session_state:
            record["kmeans_runtime_seconds"] = st.session_state["km_info"]["runtime_seconds"]
            record["kmeans_n_clusters"] = st.session_state["km_info"]["n_clusters_actual"]
            km_m = st.session_state["km_metrics"]
            record.update({"kmeans_ari": km_m["ari"], "kmeans_nmi": km_m["nmi"],
                           "kmeans_silhouette": km_m["silhouette"]})
        if "hd_info" in st.session_state:
            record["hdbscan_runtime_seconds"] = st.session_state["hd_info"]["runtime_seconds"]
            record["hdbscan_n_clusters"] = st.session_state["hd_info"]["n_clusters"]
            record["hdbscan_noise_points"] = st.session_state["hd_info"]["noise_points"]
            hd_m = st.session_state["hd_metrics"]
            record.update({"hdbscan_ari": hd_m["ari"], "hdbscan_nmi": hd_m["nmi"],
                           "hdbscan_silhouette": hd_m["silhouette"]})

        exp_id = save_experiment(record)
        st.success(f"Saved as `{exp_id}`. View on the **Experimental Results** page.")
else:
    st.caption("Run at least one method to enable saving.")
