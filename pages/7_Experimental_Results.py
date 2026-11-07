"""Page 7 — Experimental Results & Benchmarks.

Includes:
  1. Experiment History & Logs: Actual saved records, filtering, charts, and CSV export.
  2. Experiment C: Complete Graph vs. Sparse kNN Graph benchmark on manageable subsets.
"""
from __future__ import annotations

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st\n\nfrom src.ui import chart_template, init_page_ui

from src import config
from src.dense_vs_knn import run_experiment_c
from src.experiment_store import (
    clear_experiments,
    experiments_to_dataframe,
    load_experiments,
)
from src.state import get_dataset, get_embeddings

st.set_page_config(
    page_title="Experimental Results | MST Text Clustering",
    page_icon="🌳",
    layout="wide",
)\n\ninit_page_ui("Experimental Results")

st.title("Experimental Results & Benchmarks")
st.caption(
    "All charts, tables, and benchmarks use actual measurements. "
    "No values are simulated or hardcoded."
)

tab_history, tab_dense = st.tabs([
    "📊 Experiment History & Baseline Comparisons",
    "🔬 Experiment C: Complete vs. kNN Graph",
])

# ==============================================================================
# TAB 1: EXPERIMENT HISTORY
# ==============================================================================
with tab_history:
    records = load_experiments()

    if not records:
        st.info(
            "No saved experiment records yet. "
            "To record experiments, run MST algorithms and baselines on earlier pages, "
            "then click **Save this comparison to experiment log** on the **Baseline Comparison** page."
        )
    else:
        st.success(f"{len(records)} experiment record(s) loaded from `results/experiments/`.")

        # ── Filters ───────────────────────────────────────────────────────────
        st.subheader("Filter Records")
        fc1, fc2, fc3 = st.columns(3)

        k_vals = sorted({str(r.get("k", "")) for r in records if r.get("k") is not None})
        algos  = sorted({str(r.get("algorithm", "")) for r in records if r.get("algorithm")})
        sizes  = sorted({str(r.get("dataset_size", "")) for r in records if r.get("dataset_size")})

        sel_k     = fc1.multiselect("k value", k_vals, default=k_vals)
        sel_algo  = fc2.multiselect("Algorithm", algos, default=algos)
        sel_size  = fc3.multiselect("Dataset size", sizes, default=sizes)

        filtered = [
            r for r in records
            if str(r.get("k", "")) in sel_k
            and str(r.get("algorithm", "")) in sel_algo
            and str(r.get("dataset_size", "")) in sel_size
        ]

        df_filt = experiments_to_dataframe(filtered)

        st.subheader("Experiment Table")
        st.dataframe(df_filt, hide_index=True, use_container_width=True)

        # ── Export ─────────────────────────────────────────────────────────────
        csv_bytes = df_filt.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download filtered results as CSV",
            data=csv_bytes,
            file_name="mst_experiments.csv",
            mime="text/csv",
        )

        if filtered:
            st.divider()

            # ── Runtime & Memory charts ───────────────────────────────────────
            st.subheader("Algorithm Efficiency Across k")
            c_rt, c_mem = st.columns(2)

            mst_time_rows = [
                {"k": str(r.get("k", "")), "Algorithm": r.get("algorithm", ""),
                 "MST time (ms)": round(r.get("mst_runtime_seconds", 0) * 1000, 2)}
                for r in filtered
                if r.get("mst_runtime_seconds") is not None and r.get("k") is not None
            ]
            if mst_time_rows:
                fig_time = px.bar(
                    pd.DataFrame(mst_time_rows),
                    x="k", y="MST time (ms)", color="Algorithm",
                    barmode="group", template=chart_template(),
                    title="MST execution time (ms) by k",
                    color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
                )
                fig_time.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                c_rt.plotly_chart(fig_time, use_container_width=True)

            mem_rows = [
                {"k": str(r.get("k", "")), "Algorithm": r.get("algorithm", ""),
                 "Memory (MB)": round(r.get("mst_peak_memory_bytes", 0) / 1e6, 3)}
                for r in filtered
                if r.get("mst_peak_memory_bytes") is not None and r.get("k") is not None
            ]
            if mem_rows:
                fig_mem = px.bar(
                    pd.DataFrame(mem_rows),
                    x="k", y="Memory (MB)", color="Algorithm",
                    barmode="group", template=chart_template(),
                    title="Peak memory (MB) during MST execution",
                    color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
                )
                fig_mem.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                c_mem.plotly_chart(fig_mem, use_container_width=True)

            # ── Quality metrics ───────────────────────────────────────────────
            st.subheader("Clustering Quality (ARI / NMI / Silhouette)")
            qual_rows = []
            for r in filtered:
                if r.get("ari") is not None:
                    base = {"k": str(r.get("k", "")), "Algorithm": r.get("algorithm", "")}
                    qual_rows.append({**base, "Metric": "ARI",        "Score": r["ari"]})
                    if r.get("nmi") is not None:
                        qual_rows.append({**base, "Metric": "NMI",        "Score": r["nmi"]})
                    if r.get("silhouette") is not None:
                        qual_rows.append({**base, "Metric": "Silhouette", "Score": r["silhouette"]})

            if qual_rows:
                fig_qual = px.bar(
                    pd.DataFrame(qual_rows),
                    x="k", y="Score", color="Metric",
                    facet_col="Algorithm", barmode="group",
                    template=chart_template(),
                    title="Clustering quality metrics by k and algorithm",
                    color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
                )
                fig_qual.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig_qual, use_container_width=True)

            # ── MST weight verification ───────────────────────────────────────
            st.subheader("MST Total Weight by k and Algorithm")
            wt_rows = [
                {"k": str(r.get("k", "")), "Algorithm": r.get("algorithm", ""),
                 "MST weight": r.get("mst_total_weight")}
                for r in filtered
                if r.get("mst_total_weight") is not None
            ]
            if wt_rows:
                fig_wt = px.bar(
                    pd.DataFrame(wt_rows),
                    x="k", y="MST weight", color="Algorithm",
                    barmode="group", template=chart_template(),
                    title="Total MST weight (should agree across algorithms on the same graph)",
                    color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
                )
                fig_wt.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_wt, use_container_width=True)

            # ── Baseline Comparison (Latest saved run) ────────────────────────
            st.subheader("MST vs. Baselines (Latest Run)")
            last = records[-1]
            methods_in_last = []
            if last.get("ari") is not None:
                methods_in_last.append({
                    "Method": f"MST-{last.get('algorithm', '?')} (k={last.get('k', '?')})",
                    "ARI": last["ari"], "NMI": last.get("nmi"), "Silhouette": last.get("silhouette"),
                })
            if last.get("kmeans_ari") is not None:
                methods_in_last.append({
                    "Method": "K-means",
                    "ARI": last["kmeans_ari"], "NMI": last.get("kmeans_nmi"),
                    "Silhouette": last.get("kmeans_silhouette"),
                })
            if last.get("hdbscan_ari") is not None:
                methods_in_last.append({
                    "Method": "HDBSCAN",
                    "ARI": last["hdbscan_ari"], "NMI": last.get("hdbscan_nmi"),
                    "Silhouette": last.get("hdbscan_silhouette"),
                })

            if len(methods_in_last) >= 2:
                melt = pd.DataFrame(methods_in_last).melt(
                    id_vars=["Method"], value_vars=["ARI", "NMI", "Silhouette"],
                    var_name="Metric", value_name="Score",
                )
                fig_bl = px.bar(
                    melt.dropna(subset=["Score"]),
                    x="Metric", y="Score", color="Method",
                    barmode="group", template=chart_template(),
                    title="MST-based clustering vs. Independent Baselines",
                    color_discrete_sequence=["#22D3EE", "#F59E0B", "#A78BFA"],
                )
                fig_bl.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig_bl, use_container_width=True)

        st.divider()
        with st.expander("⚠️ Clear Experiment Records"):
            st.warning("This will permanently delete the JSONL experiment log.")
            if st.button("🗑 Clear all experiment records", type="secondary"):
                clear_experiments()
                st.success("Log cleared.")
                st.rerun()

# ==============================================================================
# TAB 2: EXPERIMENT C — COMPLETE VS. kNN GRAPH
# ==============================================================================
with tab_dense:
    st.subheader("Experiment C: Complete Graph vs. Sparse kNN Graph")
    st.markdown(
        "For an undirected complete graph on $n$ documents, the number of edges is "
        "$E = \\frac{n(n - 1)}{2} = O(n^2)$.  \n"
        "In contrast, a union-symmetrized kNN graph has $E \\le n \\cdot k = O(nk)$, "
        "dramatically reducing the edge count."
    )

    df_dataset, _ = get_dataset()
    embeddings_mat, _ = get_embeddings(df_dataset, _)

    c_c1, c_c2, c_c3 = st.columns(3)
    k_choice = c_c1.selectbox("k for kNN graph", config.K_VALUES, index=1, key="exp_c_k")
    algo_choice = c_c2.selectbox("MST algorithm to compare", ["Prim", "Kruskal", "Borůvka"], index=0, key="exp_c_algo")
    sizes_input = c_c3.multiselect(
        "Sample sizes to evaluate (n)",
        [500, 1000, 1500, 2000, 2500, 3000],
        default=[500, 1000, 1500, 2000],
        key="exp_c_sizes",
    )

    if st.button("▶ Run Complete vs. kNN Experiment", type="primary", key="btn_run_exp_c"):
        if not sizes_input:
            st.error("Please select at least one sample size.")
        else:
            p_bar = st.progress(0.0)
            status_text = st.empty()

            def _progress(curr, total, msg):
                p_bar.progress(curr / total)
                status_text.text(msg)

            with st.spinner("Executing benchmark across sample sizes..."):
                results_c = run_experiment_c(
                    embeddings=embeddings_mat,
                    sample_sizes=sorted(sizes_input),
                    k=int(k_choice),
                    algo_name=algo_choice,
                    repeats=3,
                    progress_callback=_progress,
                )
            p_bar.progress(1.0)
            status_text.text("Experiment C completed!")
            st.session_state["exp_c_results"] = results_c
            st.rerun()

    if "exp_c_results" in st.session_state:
        exp_c_data = st.session_state["exp_c_results"]
        st.success(f"Benchmarked {len(exp_c_data)} sample sizes!")

        # Table
        c_table = []
        for r in exp_c_data:
            c_table.append({
                "n (Docs)": r["sample_size"],
                "Complete Edges": f"{r['complete_edges']:,}",
                "kNN Edges": f"{r['knn_edges']:,}",
                "Edge Reduction": f"{r['edge_reduction_ratio']}x",
                "Complete Build (s)": f"{r['complete_build_seconds']:.3f}",
                "kNN Build (s)": f"{r['knn_build_seconds']:.3f}",
                "Complete MST (s)": f"{r['complete_mst_seconds']:.4f}",
                "kNN MST (s)": f"{r['knn_mst_seconds']:.4f}",
                "MST Speedup": f"{r['mst_speedup']}x",
                "Complete Mem (MB)": f"{r['complete_mst_memory_mb']:.2f}",
                "kNN Mem (MB)": f"{r['knn_mst_memory_mb']:.2f}",
            })
        st.dataframe(pd.DataFrame(c_table), hide_index=True, use_container_width=True)

        # Comparative Plots
        plot_df = pd.DataFrame(exp_c_data)

        # Plot 1: Edge Count Comparison (Log Scale)
        fig_edges = go.Figure()
        fig_edges.add_trace(go.Bar(
            x=[str(x) for x in plot_df["sample_size"]],
            y=plot_df["complete_edges"],
            name="Complete Graph: n(n-1)/2",
            marker_color="#F87171",
        ))
        fig_edges.add_trace(go.Bar(
            x=[str(x) for x in plot_df["sample_size"]],
            y=plot_df["knn_edges"],
            name=f"kNN Graph: k={k_choice}",
            marker_color="#22D3EE",
        ))
        fig_edges.update_layout(
            barmode="group",
            template=chart_template(),
            title="Edge Count: Complete Graph vs. Sparse kNN Graph (Log scale)",
            yaxis_type="log",
            height=380,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_edges, use_container_width=True)

        # Plot 2: MST Execution Time Comparison
        col_t1, col_t2 = st.columns(2)
        fig_mst_time = go.Figure()
        fig_mst_time.add_trace(go.Scatter(
            x=plot_df["sample_size"],
            y=plot_df["complete_mst_seconds"] * 1000,
            mode="lines+markers",
            name="Complete Graph MST",
            line=dict(color="#F87171", width=3),
        ))
        fig_mst_time.add_trace(go.Scatter(
            x=plot_df["sample_size"],
            y=plot_df["knn_mst_seconds"] * 1000,
            mode="lines+markers",
            name="kNN Graph MST",
            line=dict(color="#22D3EE", width=3),
        ))
        fig_mst_time.update_layout(
            template=chart_template(),
            title=f"{algo_choice} MST Runtime (ms)",
            xaxis_title="Sample size (n)",
            yaxis_title="Median runtime (ms)",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        col_t1.plotly_chart(fig_mst_time, use_container_width=True)

        # Plot 3: Graph Construction Time
        fig_build = go.Figure()
        fig_build.add_trace(go.Scatter(
            x=plot_df["sample_size"],
            y=plot_df["complete_build_seconds"],
            mode="lines+markers",
            name="Complete Graph Build",
            line=dict(color="#F59E0B", width=3),
        ))
        fig_build.add_trace(go.Scatter(
            x=plot_df["sample_size"],
            y=plot_df["knn_build_seconds"],
            mode="lines+markers",
            name="kNN Graph Build",
            line=dict(color="#34D399", width=3),
        ))
        fig_build.update_layout(
            template=chart_template(),
            title="Graph Construction Time (s)",
            xaxis_title="Sample size (n)",
            yaxis_title="Time (seconds)",
            height=340,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        col_t2.plotly_chart(fig_build, use_container_width=True)

        st.caption(
            "Fairness note: For an explicitly materialized complete graph, pairwise distances are "
            "computed and stored for all n(n-1)/2 pairs. For kNN, only top-k selections are kept, "
            "yielding dramatic reductions in memory and MST runtime as n scales."
        )
    else:
        st.info("Click the button above to run Experiment C.")
