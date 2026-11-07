# MST-Based Clustering of Text Embeddings
### Design and Analysis of Algorithms (DAA) Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/tests-102%20passed-brightgreen.svg)](tests/)

---

## 1. Project Overview & Research Question

Single-linkage hierarchical clustering can be constructed using a Minimum Spanning Tree (MST), where removing selected high-weight edges produces clusters of arbitrary shapes in embedding spaces. However, constructing an MST on a dense, complete similarity graph requires $\mathcal{O}(n^2)$ time and storage, making it computationally heavy for large corpora.

**Research Question:**  
*How do Prim's, Kruskal's, and Borůvka's algorithms compare when constructing Minimum Spanning Trees (or Forests) on sparse kNN graphs of text embeddings, and how do the resulting clusters compare with K-means and HDBSCAN in efficiency and clustering quality?*

### Key Features
- **End-to-End Pipeline**: Text Preprocessing $\to$ Sentence Transformers $\to$ Exact Cosine $k\text{NN}$ Graph $\to$ MST Algorithms from scratch $\to$ Size-Constrained Edge-Cut $\to$ Baseline Comparison.
- **Three Custom MST Algorithms**: Implemented from scratch (no library shortcuts) with deterministic tie-breaking `(weight, edge_id)` and minimum spanning forest support for disconnected components.
- **Independent Validation**: Recomputed total weights validated against `scipy.sparse.csgraph.minimum_spanning_tree`.
- **Size-Constrained Edge-Cut Heuristic**: Cuts the heaviest edges while guaranteeing that both resulting components contain at least `min_size` documents, preventing degenerate singleton clusters.
- **Fair Baseline Comparison**: K-means and HDBSCAN run directly on the same embeddings. Ground-truth topic labels are used **only** for evaluation metrics (Adjusted Rand Index, Normalized Mutual Information, Silhouette Score).
- **Experiment C (Complete vs. Sparse)**: Empirical evaluation demonstrating edge reduction from $\mathcal{O}(n^2)$ to $\mathcal{O}(nk)$ and runtime speedups.

---

## 2. Pipeline Architecture

```
                                  [ AG News Dataset ]
                                           │
                                           ▼ (Balanced sample: 750/class = 3,000 docs)
                                  [ Preprocessing ]
                                           │
                                           ▼
                            [ Embeddings (all-MiniLM-L6-v2) ]
                                           │
                  ┌────────────────────────┴────────────────────────┐
                  ▼                                                 ▼
        [ kNN Graph Builder ]                            [ Independent Baselines ]
        • k ∈ {5, 10, 15, 20}                            • K-means (k=4)
        • Exact cosine distance                          • HDBSCAN (density-based)
        • Union symmetrization                                      │
                  │                                                 │
                  ▼                                                 │
        [ MST Algorithms (From Scratch) ]                           │
        • Prim (Binary Heap)                                        │
        • Kruskal (DSU with size + compression)                     │
        • Borůvka (O(log V) contraction phases)                     │
                  │                                                 │
                  ▼                                                 │
        [ Size-Constrained Edge-Cutting ]                           │
        • Cuts heaviest bridge edges                                │
        • Constraint: both subtrees ≥ min_size                      │
                  │                                                 │
                  ▼                                                 ▼
        [ MST Clusters ]                                  [ Baseline Clusters ]
                  └────────────────────────┬────────────────────────┘
                                           ▼
                                [ Evaluation & Logging ]
                                • ARI, NMI, Silhouette
                                • Runtime (ms) & Memory (MB)
                                • Experiment Log (JSONL) & CSV Export
```

---

## 3. Directory Structure

```text
DAA-MST-Clustering/
├── app.py                      # Main academic dashboard & pipeline tracker
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation
├── pytest.ini                  # Pytest configuration
│
├── .streamlit/
│   └── config.toml             # Dark navy (#0B1220) and cyan (#22D3EE) styling
│
├── pages/
│   ├── 1_Dataset.py            # Balanced AG News loader & preprocessing
│   ├── 2_Embeddings.py         # all-MiniLM-L6-v2 generation & cache manager
│   ├── 3_kNN_Graph.py          # Exact kNN construction & neighbourhood explorer
│   ├── 4_MST_Algorithms.py     # Prim, Kruskal, Borůvka benchmarking & animation
│   ├── 5_Clustering.py         # Size-constrained edge-cut clustering
│   ├── 6_Baseline_Comparison.py# K-means & HDBSCAN comparison
│   ├── 7_Experimental_Results.py# Results dashboard, charts, CSV export & Exp C
│   └── 8_Methodology.py        # Detailed theoretical foundations & limitations
│
├── src/
│   ├── config.py               # Central project constants & paths
│   ├── data_loader.py          # AG News download, sampling, fingerprinting
│   ├── preprocessing.py        # Text normalisation & sanitisation
│   ├── embeddings.py           # SentenceTransformer wrapper, L2-norm, caching
│   ├── knn_graph.py            # Cosine kNN search, union symmetrisation, Graph
│   ├── mst.py                  # Custom Prim, Kruskal, Borůvka implementations
│   ├── mst_validation.py       # Independent cycle, edge count & SciPy validation
│   ├── benchmark.py            # Consistent timing (perf_counter) & memory (tracemalloc)
│   ├── clustering.py           # Size-constrained MST edge-cut heuristic
│   ├── baselines.py            # Scikit-learn K-means and HDBSCAN wrappers
│   ├── evaluation.py           # ARI, NMI, and Cosine Silhouette scoring
│   ├── dense_vs_knn.py         # Complete graph vs kNN graph (Experiment C)
│   ├── experiment_store.py     # Append-only JSONL experiment store
│   ├── state.py                # Streamlit session state managers
│   └── visuals.py              # Plotly graph network figure helpers
│
├── tests/
│   ├── test_data.py            # Preprocessing & sampling unit tests
│   ├── test_embeddings.py      # Normalisation & cache invalidation tests
│   ├── test_knn_graph.py       # Exact kNN & symmetrisation tests
│   ├── test_mst.py             # MST correctness & tie-breaking tests
│   ├── test_clustering.py      # Size-constrained edge-cut tests
│   ├── test_evaluation.py      # Metric computation tests
│   └── test_dense_vs_knn.py    # Complete graph vs kNN tests
│
├── data/                       # Local data storage (Git-ignored)
│   ├── processed/              # ag_news_subset.csv & metadata
│   ├── embeddings/             # embeddings.npz & metadata
│   └── graphs/                 # knn_k*.npz & metadata
│
└── results/                    # Experimental outputs
    ├── experiments/            # experiment_log.jsonl
    ├── charts/
    └── reports/
```

---

## 4. Setup & Installation (Windows 10 / 11)

### Prerequisites
- Python 3.11 or higher (64-bit recommended).

### 1. Create and Activate Virtual Environment
Open PowerShell inside the project directory:

```powershell
# Navigate to project folder
cd "d:\Onedrive SRM\OneDrive - srmap.edu.in\SRM AP PDF\MY WORK\DAA Project\DAA-MST-Clustering"

# Create virtual environment
py -3.11 -m venv .venv
# or: python -m venv .venv

# Activate environment
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. Running the Application

Launch the Streamlit interactive dashboard:

```powershell
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Step-by-Step Execution Guide:
1. **1_Dataset**: Click *"Load AG News and create balanced subset"*. A balanced subset of 3,000 documents (750 per topic) with fixed random seed 42 will be created and saved.
2. **2_Embeddings**: Click *"Generate embeddings"*. Texts are encoded with `all-MiniLM-L6-v2` into 384-dimensional unit vectors. Embeddings are cached to disk so they never need to be recomputed.
3. **3_kNN_Graph**: Click *"Build all four graphs (k = 5, 10, 15, 20)"*. Exact cosine neighbours are found and union-symmetrized.
4. **4_MST_Algorithms**: Select a graph (e.g., $k=15$) and click *"Run all three"*. Prim, Kruskal, and Borůvka execute on the exact same read-only graph arrays and validate against SciPy.
5. **5_Clustering**: Set target clusters ($4$) and minimum cluster size ($10$), then click *"Run edge-cut clustering"*. The heaviest edges are cut if both subtrees meet the minimum-size condition.
6. **6_Baseline_Comparison**: Run K-means and HDBSCAN directly on the embeddings, compare ARI/NMI/Silhouette side by side, and save the experiment record.
7. **7_Experimental_Results**: Explore runtime vs. $k$, memory consumption, cluster metrics, and run **Experiment C** to benchmark the complete graph vs. the sparse $k\text{NN}$ graph.

---

## 6. Running Tests

Run the comprehensive unit test suite:

```powershell
pytest -v
```

All 102 unit tests test:
- Dataset sampling balance & reproducibility
- Text cleaning and whitespace normalization
- Embedding L2-normalization and cache invalidation
- Exact $k\text{NN}$ search vs. brute-force and scikit-learn
- Prim, Kruskal, and Borůvka edge selections, cycle prevention, and tie-breaking
- Minimum spanning forest handling on disconnected graphs
- Size-constrained edge-cutting heuristic constraints
- ARI, NMI, and Silhouette calculations with noise exclusion
- Complete graph materialization and Experiment C comparison

---

## 7. Algorithms & Time Complexity Summary

| Algorithm | Data Structure | Time Complexity | Memory | Disconnected Graphs |
|---|---|---|---|---|
| **Prim** | Binary Min-Heap + Adjacency List | $\mathcal{O}((V + E) \log V)$ | $\mathcal{O}(V + E)$ | Minimum Spanning Forest (restarts per component) |
| **Kruskal** | Disjoint Set Union (DSU with union-by-size & path compression) | $\mathcal{O}(E \log E + E \cdot \alpha(V))$ | $\mathcal{O}(V + E)$ | Minimum Spanning Forest (natural) |
| **Borůvka** | Component array + parallel edge scan | $\mathcal{O}(E \log V)$ in $\le \lceil \log_2 V \rceil$ phases | $\mathcal{O}(V + E)$ | Minimum Spanning Forest (natural) |
| **Complete Graph MST** | All $E = \frac{n(n-1)}{2}$ edges | $\mathcal{O}(n^2 \log n)$ | $\mathcal{O}(n^2)$ | Always connected if weights finite |
| **$k\text{NN}$ Graph MST** | Sparse $E \le n \cdot k$ edges | $\mathcal{O}(nk \log(nk))$ | $\mathcal{O}(nk)$ | Spanning Forest for $k < k_{\text{threshold}}$ |

---

## 8. Academic Integrity & Technical Principles

1. **No Label Leaking**: Ground-truth class labels are never provided to embedding generation, graph construction, MST algorithms, or edge-cut heuristics. They are solely referenced when calculating evaluation metrics (ARI / NMI).
2. **Deterministic Tie-Breaking**: All edges are compared by `(weight, edge_id)`. This establishes a strict total order, guaranteeing unique minimum spanning trees for connected graphs with unique edge weights.
3. **No Fabricated Data**: All dashboard metrics, runtime numbers, and evaluation scores are computed live from user actions or loaded directly from disk-persisted runs.
4. **Independent Baselines**: K-means and HDBSCAN consume the embedding matrix directly and do not rely on the custom $k\text{NN}$ graph or MST.
