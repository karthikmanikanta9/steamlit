"""Page 9 — Methodology / About.

Static explanation of the research question, design decisions,
and experimental limitations.
"""
import streamlit as st

st.set_page_config(
    page_title="Methodology | MST Text Clustering",
    page_icon="🌳",
    layout="wide",
)

st.title("Methodology & About")
st.caption("Design and Analysis of Algorithms — MST-Based Clustering of Text Embeddings")

st.markdown("""
## Research Question

> How do **Prim's**, **Kruskal's**, and **Borůvka's** algorithms compare when constructing
> MSTs on kNN graphs of text embeddings, and how do the resulting clusters compare with
> **K-means** and **HDBSCAN** in efficiency and clustering quality?
""")

st.divider()

st.subheader("Pipeline Overview")
st.markdown("""
```
Documents  →  Embeddings  →  kNN Graph  →  MST / MSF  →  Edge-cut  →  Clusters
                                                              ↕
                                                       K-means / HDBSCAN
                                                      (independent baselines)
```

Ground-truth topic labels are **never** used to construct embeddings, the kNN graph, or the MST.
They are retained solely for evaluation (ARI, NMI).
""")

st.divider()

st.subheader("1. Dataset")
st.markdown("""
- **Source:** AG News (via Hugging Face `fancyzhx/ag_news`)
- **Classes:** World, Sports, Business, Sci/Tech
- **Subset:** 750 documents per class = **3,000 documents** total (balanced)
- **Seed:** fixed (42) for reproducibility — same seed produces an identical fingerprint
- **Preprocessing:** whitespace normalisation; empty/invalid records removed;
  document IDs stabilised; no aggressive stop-word removal (inappropriate for
  sentence-transformer models)
""")

st.subheader("2. Embeddings")
st.markdown("""
- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimension:** 384
- **Normalisation:** L2 (unit-length vectors)
- **Device:** CPU
- Embeddings are generated **once** and cached. They are reloaded on subsequent runs
  without regeneration, keyed by a SHA-256 fingerprint of the dataset subset.
""")

st.subheader("3. Distance Metric")
st.markdown(r"""
$$d(u,v) = 1 - \cos(u,v) = 1 - \frac{u \cdot v}{\|u\| \, \|v\|}$$

Because all vectors are L2-normalised, $d(u,v) = 1 - u \cdot v$.
Smaller distance means greater semantic similarity.
""")

st.subheader("4. kNN Graph Construction")
st.markdown("""
For each of **k = 5, 10, 15, 20**:

1. Find each document's *k* nearest neighbours by cosine distance (exact search).
2. Apply **union symmetrisation**: retain edge {i, j} if *i* chose *j* **or** *j* chose *i*.
3. Remove duplicate edges and self-loops.
4. Each undirected pair gets exactly one weight (the computed cosine distance).
5. Graph statistics and connected-component count are computed and displayed.

A disconnected kNN graph has no spanning tree; it has a **minimum spanning forest**.
No edges are added to bridge separate components.
""")

st.subheader("5. MST Algorithms (implemented from scratch)")

with st.expander("Prim's Algorithm"):
    st.markdown("""
- Grows a single tree from a chosen start vertex.
- Uses a **binary heap (priority queue)** over candidate edges.
- In forest mode: restarts in each disconnected component.
- Complexity: O((V + E) log V) with an adjacency list + heap.
    """)

with st.expander("Kruskal's Algorithm"):
    st.markdown("""
- Sorts all edges by weight (O(E log E)).
- Uses **Disjoint Set Union (Union-Find)** with path compression and union by size.
- Adds an edge only if it joins two different components.
- Naturally produces a minimum spanning forest for disconnected graphs.
- Complexity: O(E log E + E · α(V)), where α is the inverse Ackermann function.
    """)

with st.expander("Borůvka's Algorithm"):
    st.markdown("""
- Initially treats every vertex as its own component.
- Each **phase**: every component selects its cheapest outgoing edge →
  all such edges are added → components merge.
- The number of components **at least halves** each phase → O(log V) phases.
- Complexity: O(E log V).
- Phase count is recorded and displayed.
    """)

st.markdown("""
**Deterministic tie-breaking:** edges are compared by `(weight, edge_id)`.
Because edge IDs are unique, this yields a strict total order.
All three algorithms are guaranteed to agree on the **total MST weight**
for the same connected graph. They may select different valid edge sets when
weights are equal, which is correct behaviour.
""")

st.subheader("6. MST Validation")
st.markdown("""
Every MST/MSF result is validated against an **independent SciPy reference**:

| Check | Description |
|---|---|
| Edge count | V − 1 for a connected graph; V − C for C components |
| No cycles | Union-Find confirms no cycle is introduced |
| Spans all components | One tree per connected component |
| Weight consistency | Recomputed weight agrees with reported weight |
| SciPy reference | Total weight matches `scipy.sparse.csgraph.minimum_spanning_tree` |
""")

st.subheader("7. Edge-Cut Clustering")
st.markdown("""
The MST does **not** determine the number of clusters by itself.
The **size-constrained heuristic** does:

1. Sort MST edges from **heaviest to lightest** (candidates to cut).
2. For each candidate: remove it *only if* both resulting components have
   ≥ `min_size` documents.
3. Stop when `target_k` clusters are reached, or no eligible edge remains.
4. Report the actual cluster count and explain any shortfall.

**This heuristic does not guarantee exactly 4 clusters.**
If the size constraint prevents further cuts, fewer clusters are produced and explained.

For disconnected spanning forests, each initial tree counts as a separate starting cluster.
""")

st.subheader("8. Baselines")
st.markdown("""
Both baselines operate **directly on the embedding matrix**.
They do **not** use the kNN graph or project MST.

| Method | Input | Key parameters |
|---|---|---|
| K-means | Embedding matrix | k (clusters), seed, n_init |
| HDBSCAN | Embedding matrix | min_cluster_size, min_samples |

HDBSCAN may internally construct an MST as part of its own algorithm.
It is still an independent baseline here — it does not consume the Phase 5 MST.

**Noise points** (HDBSCAN label −1) are excluded from ARI, NMI, and silhouette.
""")

st.subheader("9. Evaluation Metrics")
st.markdown(r"""
| Metric | Description | Range |
|---|---|---|
| **ARI** | Adjusted Rand Index — agreement between predicted and true labels, adjusted for chance | −1 to 1 (1 = perfect) |
| **NMI** | Normalized Mutual Information — shared information between predicted and true labels | 0 to 1 (1 = perfect) |
| **Silhouette** | Cohesion vs. separation using cosine distance; singleton clusters excluded | −1 to 1 (1 = perfect) |

Ground-truth labels are **only used here** (evaluation). They are never used upstream.
""")

st.subheader("10. Experimental Limitations")
st.markdown("""
- All computation is on **CPU**; a GPU would accelerate embedding generation.
- The exact kNN search is O(n²) in memory for large n; FAISS or ANNOY would
  reduce this for larger datasets.
- Silhouette is subsampled (up to 2,000 points) to avoid O(n²) cost.
- The edge-cut heuristic is greedy and not globally optimal.
- kNN graph sparsity can produce disconnected graphs at low k,
  which prevents a single spanning tree.
- Experiment records are stored locally; multi-user or cloud scenarios would
  require a database.
""")

st.divider()
st.markdown("""
**Project:** MST-Based Clustering of Text Embeddings  
**Course:** Design and Analysis of Algorithms (DAA)  
**Stack:** Python · Streamlit · scikit-learn · sentence-transformers · SciPy · Plotly
""")
