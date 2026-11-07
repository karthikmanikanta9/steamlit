# Baseline Audit — MST-Based Clustering of Text Embeddings

**Repository:** `karthikmanikanta9/steamlit`  
**Baseline commit:** `63cb683357ae16680f8063e330a7171169110e62`  
**Working branch:** `audit/phase-1-9-remediation`  
**Audit started:** 2026-11-07  
**Purpose:** Record the existing project's state before changing algorithm behaviour or redesigning pages.

---

## 1. Repository inventory

The baseline repository contains:
- Streamlit application entry point: `app.py`
- Eight Streamlit pages: Dataset, Embeddings, kNN Graph, MST Algorithms, Clustering, Baseline Comparison, Experimental Results, Methodology
- Core modules for configuration, data preparation, embeddings, graph construction, MST algorithms, clustering, baselines, evaluation, benchmarking, storage, and visualizations
- Seven existing test modules
- One checked-in JSONL experiment record
- A README and dark navy/cyan Streamlit theme

The actual source tree is substantially beyond the original reported Phase 3 status: application code for all project stages exists. This audit treats those stages as implemented-but-not-yet-fully-verified until their correctness and runtime gates pass.

## 2. Automated baseline result

### First baseline run — passed

GitHub Actions run: [Python 3.14 baseline run](https://github.com/karthikmanikanta9/steamlit/actions/runs/42346428052)

| Check | Result |
|---|---|
| Python runtime | 3.14.10 |
| Application/source/page compilation | Passed |
| Existing pytest suite | **102 passed in 2.48 seconds** |
| Workflow job | Success |

The CI runner installed test dependencies (`numpy`, `pandas`, `scipy`, `scikit-learn`, and `pytest`). This first baseline did **not** launch Streamlit, download the sentence-transformer model, or verify the user's local data/embedding cache.

### Expanded CI workflow

The test workflow has since been extended to:
- use Node 24-compatible GitHub Actions releases;
- install Streamlit and Plotly for a dashboard entry-point smoke test;
- compile the source and pages;
- rerun all tests;
- upload test output artifacts.

The expanded run must pass before its dashboard smoke check is considered verified. Do not treat the earlier 102-test result as proof that this new smoke test passed.

## 3. Local verification still required

GitHub Actions runs on Linux and cannot prove the complete application behaves identically on the user's Windows 10 / Python 3.14 machine.

The following still require local verification:
- exact Python version and installed dependency versions;
- test results in the active Windows virtual environment;
- Streamlit app startup;
- loading the user's ignored local dataset and embedding cache;
- graph construction and MST execution using those local artifacts;
- the rendered pages and plots in the user's browser.

The directories `data/processed/`, `data/embeddings/`, and `data/graphs/` are ignored by Git. That is intentional for large generated artifacts, but their actual contents were not available to this repository audit.

## 4. Static findings to resolve

The following issues were identified by source inspection. The list is a remediation backlog, not a claim that no other defects exist.

### High priority — correctness and evaluation

1. **Edge-cut target status — `src/clustering.py`**  
   The current flag uses `actual_k >= target_k`. A forest that begins with more components than the requested target can be reported as reaching the target even though the exact requested cluster count was not produced. Define exact target attainment, overshoot, and unattainable-target states separately.

2. **Single-cluster ARI/NMI — `src/evaluation.py`**  
   ARI and NMI are mathematically defined for a one-cluster predicted partition. The current code returns `None` when fewer than two predicted labels remain. Return a score for non-empty evaluated sets; use `None` only when the metric is genuinely undefined (for example, no evaluated samples).

3. **HDBSCAN noise protocol — `src/evaluation.py`, `src/baselines.py`, baseline page**  
   Current HDBSCAN evaluation drops label `-1` rows, while MST and K-means are scored across the entire dataset. Keep clustered-only scores, but show their coverage/sample count and add a clearly named all-document view treating noise as its own predicted label. Do not present unlike sample populations as directly comparable.

4. **Experiment C sample selection — `src/dense_vs_knn.py`**  
   The current implementation takes `embeddings[:sample_size]`. Because the balanced subset is sorted by document ID, a prefix is not guaranteed to represent all four classes. Use a deterministic documented index-selection policy and exactly the same rows for complete and kNN graph construction at every sample size.

5. **K-means geometry description — `app.py`, `src/baselines.py`, README/methodology**  
   The UI describes K-means as using cosine geometry, but calls standard scikit-learn `KMeans`, which optimizes the Euclidean objective. Either label the existing baseline accurately as Euclidean K-means on normalized embeddings or implement a separately named, tested spherical/cosine K-means variant.

6. **Neighbour tie determinism — `src/knn_graph.py`**  
   A stable sort after `argpartition` cannot recover deterministic ordering among candidates tied at the k-th selection boundary. Make the selection rule deterministic before finalizing the top-k set, for example similarity descending then vertex index ascending, and test exact ties.

### High priority — artifact integrity and stale state

7. **Graph and cache validation — `src/knn_graph.py`, `src/embeddings.py`, `src/data_loader.py`**  
   Strengthen validation of array lengths/shapes, vertex bounds, finite edge weights, graph metadata, matrix finiteness/norms, dataset fingerprint, model identity, document IDs, and corrupt/truncated files. Cache load failures should become clear recoverable UI states.

8. **Downstream state invalidation — `src/state.py`, Streamlit pages**  
   Dataset or embedding changes must invalidate dependent graphs, MST results, edge-cut labels, baseline labels/metrics, and page selections. Rebuilding one kNN graph must invalidate only the MST/clusters derived from that graph. Do not show old session-state results as belonging to new inputs.

9. **MST result validation — `src/mst_validation.py`, `src/mst.py`**  
   Add checks for result vertex count, invalid graph/edge inputs, non-finite weights, and Prim start-vertex bounds. Preserve the independent SciPy reference and the (V-C) forest edge-count check.

### Medium priority — benchmarking and experiment provenance

10. **Memory benchmark terminology — `src/benchmark.py`, `src/baselines.py`**  
    `tracemalloc` reports Python-tracked allocations; it is not a complete measure of process RSS or native-library allocations. Label it accurately and add process-level memory measurement where it is reliable, otherwise disclose the limitation.

11. **Experiment provenance and malformed logs — `src/experiment_store.py`, result page**  
    Store dataset/embedding fingerprints, sample IDs or sampling seed, model/backend/library versions, graph settings, algorithm parameters, metric protocol, repeat counts, and code revision. Do not silently discard malformed JSONL lines; expose a warning/count to the user.

12. **Latest/filtered experiment consistency — `pages/6_Baseline_Comparison.py`, `pages/7_Experimental_Results.py`**  
    Select the newest result by an explicit timestamp/ID, not incidental dictionary insertion order. In the results page, calculate the “latest” comparison from the filtered records when filters are active, not always from the global last record.

13. **Dense-graph resource use — `src/dense_vs_knn.py`, results page**  
    Complete graph creation materializes (n(n-1)/2) edge pairs and currently uses several arrays/copies. Add a resource estimate and confirmation, minimize duplicate allocations, and make maximum sample-size guidance reflect the user's 12 GB system.

14. **Input and edge cases**  
    Add explicit behaviour/tests for empty text lists, invalid batch/chunk sizes, zero-vertex graphs, isolated vertices, invalid k, malformed DataFrames, invalid metric sample sizes, and mismatched labels/embedding rows.

### Medium priority — tests and documentation

15. **Clustering test strength — `tests/test_clustering.py`**  
    Some current assertions can pass without proving every output cluster satisfies the minimum-size invariant. Replace permissive conditions with exact hand-computed expectations and independent invariant checks.

16. **Baseline/integration coverage**  
    Add dedicated tests for K-means/HDBSCAN wrappers (using deterministic synthetic fixtures or test doubles), evaluation protocols, artifact invalidation, and an end-to-end pipeline run using small synthetic data.

17. **README and methodology accuracy**  
    Align Python setup commands with Python 3.14 support, correct K-means and tie-breaking descriptions, distinguish exact-neighbour computation work from sparse retained-edge count, and keep the advertised test count tied to CI evidence.

## 5. What this audit does not claim

- The existing app has not yet been fully exercised on the user's Windows machine.
- The sentence-transformer model was not downloaded or run in the GitHub baseline job.
- Ignored local data, embeddings, and graph caches were not inspected by the repository connector.
- A passing existing test suite does not prove that untested requirements or all page interactions are correct.
- The static findings above are not an exhaustive proof that every bug has been found; subsequent phases include targeted tests and broader code review.

## 6. Next phase

Proceed to **Phase B — Data, embeddings, and artifact integrity** only after the user has checked out this working branch locally and reported the Windows/Python 3.14 baseline test result. No application algorithm changes are part of this baseline audit.
