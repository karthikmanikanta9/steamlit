# Implementation & Remediation Plan
## MST-Based Clustering of Text Embeddings

**Repository:** `karthikmanikanta9/steamlit`  
**Working branch:** `audit/phase-1-9-remediation`  
**Baseline:** current `main` commit `63cb683357ae16680f8063e330a7171169110e62`  
**Runtime target:** Windows 10/11, Python 3.14, CPU-first  
**Application:** Streamlit

---

## 1. Purpose and working rules

The repository already contains implementations for the dataset, embeddings, kNN graph, all three MST algorithms, MST edge-cut clustering, K-means/HDBSCAN baselines, evaluation, experiment logging, Experiment C, and eight Streamlit pages. The objective is therefore to **audit and correct the existing project**, not replace it with a different stack or rewrite working components without a reason.

The current `main` branch is the untouched reference. All remediation work happens on this branch and should be delivered in small, reviewable commits or pull requests.

Rules for every phase:

1. Preserve the current Streamlit + modular Python architecture unless a specific technical reason justifies a change.
2. Do not use ground-truth topic labels as input to embeddings, graph construction, MST algorithms, edge cutting, K-means, or HDBSCAN. Labels may be used for transparent stratified experiment sampling only when that sampling choice is explicitly documented; labels remain evaluation metadata for the clustering methods.
3. Never fabricate runtime, memory, graph, or quality values. Every displayed result must come from a real run or be clearly marked as an illustrative example.
4. Every algorithm change must include tests for normal cases, edge cases, invalid inputs, and disconnected graphs where applicable.
5. Do not merge changes into `main` until the phase's tests and review gate pass.
6. Generated datasets, embeddings, graphs, and model caches remain outside Git. Their metadata, fingerprints, configuration, and instructions must make local reproduction possible.

---

## 2. Findings to resolve

The initial repository inspection identified these priority issues. They are starting findings, not a claim that every possible defect has already been exhausted; each phase includes a broader review.

| Area | File(s) | Finding / risk |
|---|---|---|
| Edge-cut target reporting | `src/clustering.py` | `reached_target` uses `actual_k >= target_k`. An initial forest with more components than the target can be incorrectly reported as having reached the target. Exact target attainment and impossible-target cases must be distinguished. |
| ARI/NMI | `src/evaluation.py` | The implementation returns `None` for fewer than two predicted clusters. ARI and NMI are defined for a one-cluster prediction and should be computed when there are evaluated samples. |
| HDBSCAN evaluation | `src/evaluation.py`, `src/baselines.py`, baseline page | HDBSCAN noise points are removed from scores, while other methods are scored on all documents. This is useful as a clustered-only score, but is not a like-for-like comparison unless coverage and a full-set noise-as-a-cluster score are also reported. |
| Experiment C sampling | `src/dense_vs_knn.py` | Taking `embeddings[:sample_size]` does not guarantee representative or balanced prefixes. The same deterministic index set must be used for both complete and kNN graphs at each n. |
| K-means geometry description | `app.py`, `src/baselines.py`, documentation | The dashboard describes cosine geometry, but the implementation calls standard scikit-learn KMeans, whose objective is Euclidean. Documentation and the selected baseline must agree. |
| kNN tie determinism | `src/knn_graph.py` | A stable sort after `argpartition` does not make selection deterministic when ties occur at the k-th boundary. Define and test a deterministic secondary ordering. |
| Artifact validation | `src/embeddings.py`, `src/knn_graph.py`, `src/data_loader.py` | Cache reads need stronger corruption, shape, ID, finite-value, normalization, metadata, and lineage checks. |
| Session-state invalidation | `src/state.py`, dataset/embedding/graph pages | Changing an upstream artifact must invalidate all dependent graph, MST, clustering, baseline, and displayed result state. |
| Benchmark semantics | `src/benchmark.py`, `src/baselines.py` | `tracemalloc` is not total process RAM, especially for NumPy/SciPy/native allocations. Label it accurately and add an OS-process memory measure where feasible. |
| Test strength | `tests/test_clustering.py` and missing baseline/integration tests | Some assertions are too permissive; dedicated baseline, artifact-integrity, state-invalidation, and end-to-end coverage are needed. |
| Reproducibility | `src/experiment_store.py`, results pages | Experiment records need dataset/model/config/version fingerprints and complete parameter sets. Malformed records should be visible as warnings rather than silently disappearing. |
| CI | repository workflows | No GitHub Actions workflow runs were present at the time of inspection. Add automated testing for the target Python version. |

The current repository advertises 102 passing tests. That number is a repository claim, not a result independently run during the GitHub inspection. It must be re-established by a clean test run and CI.

---

## 3. Remediation phases

### Phase A — Establish a reliable baseline and audit inventory

**Goal:** Know exactly what is currently working before changing code.

Tasks:
- Inventory all tracked files and reconcile the README, actual tree, imports, and page navigation.
- Confirm Python 3.14 and exact installed package versions on the user's machine.
- Run the current complete test suite and save the unedited output.
- Record current Streamlit startup behaviour and page-by-page smoke-test status.
- Record current artifact presence locally: dataset CSV/metadata, embedding NPZ/metadata, graph caches, and experiment log.
- Verify that no secrets, personal tokens, private files, virtual environments, or unintended large files are tracked.
- Establish a baseline test count and identify tests that are missing, weak, skipped, or flaky.

**Acceptance gate:**
- Baseline test output is captured.
- Every tracked application and test file is inventoried.
- Missing local artifacts and unverified behaviours are explicitly recorded.
- No code changes are mixed into this baseline commit.

### Phase B — Data, embeddings, and artifact integrity

**Goal:** Every downstream stage must consume the exact dataset and embedding matrix it claims to consume.

Tasks:
- Recompute and validate the saved dataset fingerprint when loading the dataset, not only when initially creating it.
- Validate required columns, row count, unique document IDs, valid labels, nonempty text, label-name alignment, and metadata consistency.
- Strengthen embedding cache validation: model ID, dataset fingerprint, ordered document IDs, matrix dimensions, row count, dtype, finite values, and unit norms.
- Handle missing, truncated, malformed, or mismatched NPZ/JSON files with a clear recoverable message.
- Make paired artifact writes safer (temporary files followed by replacement) so an interrupted save does not leave a seemingly valid half-cache.
- Handle empty inputs and invalid matrix shapes explicitly.
- Ensure a dataset replacement invalidates every dependent artifact and in-memory result.

**Acceptance gate:**
- Cache hit, cache miss, stale dataset, stale model, reordered IDs, corrupted metadata, corrupted NPZ, NaN/Inf values, wrong norms, and wrong dimensions are tested.
- No stale embeddings can be used with a different dataset fingerprint.

### Phase C — kNN graph correctness and reproducibility

**Goal:** Build a deterministic, well-formed weighted undirected graph from normalized embeddings.

Tasks:
- Validate graph inputs: n, k, embedding dimensions, finite values, row norms, neighbour index bounds, edge-array lengths, self-loops, duplicate pairs, and weights.
- Define deterministic top-k ordering including exact and near ties at the selection boundary (e.g. similarity descending, document index ascending).
- Preserve the documented union symmetrization rule: keep an undirected pair if either endpoint selects the other.
- Verify that each undirected edge has one cosine-distance weight, with (0 leq d leq 2).
- Validate graph-cache metadata against the source embedding fingerprint, embedding generation, k, graph method, and code/schema version.
- Test graph statistics for isolated vertices and disconnected graphs.
- Clearly document exact-search cost: exact all-pairs neighbour search performs quadratic pairwise similarity work; chunking bounds the temporary similarity matrix, while the retained graph has at most (n k) undirected edges under this construction.

**Acceptance gate:**
- Deterministic repeated construction on tie-heavy data.
- Exact neighbour results agree with an independent brute-force oracle under the defined tie rule.
- No loops or duplicate edges; valid symmetric undirected representation; correct components and weights.
- Cache round-trip and stale-cache rejection tests pass.

### Phase D — MST algorithms, validation, and benchmarking

**Goal:** Prove Prim, Kruskal, and Borůvka return valid minimum spanning trees/forests on the same graph.

Tasks:
- Audit the implementations and input validation for empty graphs, single vertices, disconnected graphs, invalid start vertices, tied weights, isolated vertices, and non-finite or invalid edge weights.
- Keep algorithms independent and implemented in project code; use SciPy only as an independent validation oracle.
- Define a strict deterministic edge order, such as `(weight, edge_id)`, and apply it consistently in every algorithm.
- Validate edge IDs, uniqueness, edge count (V-C), acyclicity, spanning of every original graph component, reported weight, and independent reference weight.
- Revisit the claim that identical edge sets are expected: only make that claim when the strict total ordering is actually enforced by all implementations and verified.
- Separate timing runs from memory runs; report median and spread, repeat count, algorithm setup included/excluded, and measurement method.
- Keep `tracemalloc` labelled as Python-tracked allocation. Add a process-RSS/working-set measurement or explicitly report that total peak process memory is not available on the current setup.

**Acceptance gate:**
- Deterministic textbook, random, tie-heavy, disconnected, isolated-vertex, and real-kNN tests pass.
- Independent reference weight agrees within a documented tolerance.
- Benchmark labels accurately describe the quantity measured.

### Phase E — MST edge-cut clustering correctness

**Goal:** Produce defensible clusters from the MST/MSF and report constraints truthfully.

Tasks:
- Validate that the supplied MST result belongs to the supplied graph and is a valid forest before cutting it.
- Correct target reporting: distinguish exactly reached, fewer than target, and more than target because the input forest already starts with too many components.
- Reject invalid targets and minimum sizes with clear messages.
- Guarantee every created component satisfies the stated minimum-size rule.
- Report initial forest components, cuts, final cluster count, all component sizes, skipped candidates, and why the requested target may be impossible.
- Strengthen tests so they independently assert every cluster size and exact expected results; remove permissive assertions that can pass without proving the intended property.
- Profile the repeated component traversals and improve them if larger graphs make the current implementation too slow, without sacrificing correctness.

**Acceptance gate:**
- Exact hand-calculated cases pass.
- Disconnected forest, target below initial component count, target equal to initial count, unattainable target, minimum-size boundary, and invalid MST input are tested.
- The UI never says a target was reached unless the defined target condition is actually satisfied.

### Phase F — Evaluation protocol and baseline correctness

**Goal:** Make MST, K-means, and HDBSCAN quality comparisons scientifically interpretable.

Tasks:
- Compute ARI and NMI for one-cluster predictions when there are evaluated samples; reserve `None` for genuinely undefined/empty cases.
- Report HDBSCAN coverage (non-noise fraction) and noise count.
- Report two clearly named views for HDBSCAN: full-dataset metrics treating noise label `-1` as an assigned cluster, and clustered-only metrics with noise excluded. Never compare the clustered-only score with an all-document score without stating the difference.
- For silhouette, document cosine distance, sample size and seed; handle one cluster, all noise, too few samples, and singleton clusters. Report how many points were excluded for noise and singleton handling.
- Validate equal lengths and row alignment across embeddings, true labels, and predicted labels.
- Resolve the K-means geometry mismatch. The default implementation is standard Euclidean K-means on L2-normalized embeddings; either document it accurately or deliberately implement and test spherical/cosine K-means as a separately named method. Do not call standard KMeans a cosine algorithm.
- Standardize timing and memory measurement descriptions across MST and baseline algorithms.
- Record exact baseline backend and package version, because scikit-learn HDBSCAN and the external `hdbscan` package can differ in implementation details.

**Acceptance gate:**
- Tests cover one cluster, two clusters, all noise, partial noise, singleton clusters, label-length mismatch, and perfect/random partitions.
- The UI shows scores and coverage together.
- Baseline method names and metric definitions match the actual implementation.

### Phase G — Experimental design, logging, and reproducibility

**Goal:** Ensure every chart and conclusion is traceable to a fair, repeatable experiment.

Tasks:
- Replace prefix sampling in Experiment C with a deterministic sample-index policy. Use identical document indices for complete and kNN graph runs at each sample size.
- Record the sampling policy and seed; if the sample is stratified, document that ground-truth labels are used only to select the experimental subset, never as clustering features or input.
- Report connected-component count and whether each result is an MST or MSF.
- Separate and measure graph construction, neighbour search, retained graph storage, MST execution, and memory. Do not claim that exact kNN search itself is (O(nk)); only the retained edge count is sparse.
- Add a memory-aware limit and a warning/confirmation before materializing large complete graphs. Use compact arrays and avoid unnecessary copies.
- Save a complete experiment manifest: dataset fingerprint, embedding model and fingerprint, k, graph symmetrization, sample IDs/seed, algorithm and parameters, baseline settings, metric protocol, package versions, timing repeats, hardware/runtime details, and code commit.
- Validate experiment record schema; expose malformed-line counts and errors rather than silently hiding them.
- Make chart filters and “latest run” logic consistent with the records selected by the user.
- Do not infer an algorithm winner from one run; show distributions or repeated-run summaries where appropriate.

**Acceptance gate:**
- Repeated runs with the same manifest reproduce the same sample IDs, graph structure (under the tie policy), and clustering configuration.
- Dense and sparse runs use the same embeddings and document indices.
- CSV/JSON exports include enough metadata to reproduce the run.
- No performance conclusion is hardcoded.

### Phase H — Application state, orchestration, and persistence

**Goal:** Make the whole application reliable across reruns, page changes, and restarts.

Tasks:
- Introduce a consistent artifact lineage/version key: dataset → embeddings → graph(k) → MST → edge-cut clusters → metrics/experiment.
- Centralize invalidation. Changing the dataset invalidates embeddings, graphs, MST results, clusters, baselines, and experiment views that depend on the previous data. Regenerating embeddings invalidates all graph and downstream results. Rebuilding a graph invalidates its MST and clustering outputs.
- Persist reusable derived artifacts where justified (graph, MST result, cluster labels/metadata) so a server restart does not silently erase expensive work. Keep large files outside Git.
- Make page prerequisites explicit and actionable.
- Add robust exception boundaries and useful error messages without swallowing programming errors.
- Keep all displayed status indicators derived from validated artifacts, not merely from the presence of a session-state key.
- Use Streamlit caching only with correct artifact fingerprints and parameters.

**Acceptance gate:**
- A page refresh/restart reloads valid artifacts.
- Upstream changes invalidate downstream results.
- No page can show stale labels or metrics attached to a different dataset/embedding/graph.
- A user can follow the workflow in the intended order without hidden session-state dependencies.

### Phase I — Full website UX, quality assurance, documentation, and release

**Goal:** Deliver a polished academic analytics website backed by the verified pipeline.

Tasks:
- Apply the UI specification in `docs/WEBSITE_DESIGN_SPEC.md`.
- Make navigation and workflow progression clear: Dashboard → Dataset → Embeddings → kNN → MST → Clustering → Baselines → Experiments → Reports/Methodology.
- Add complete loading, empty, success, warning, and error states.
- Add accessible labels, readable chart legends, meaningful units, consistent number formatting, and responsive layouts.
- Provide downloadable experiment CSV and JSON manifest, plus a concise report export with methodology and run metadata.
- Add GitHub Actions CI for the supported Python version(s), initially Python 3.14, and run the complete test suite on every pull request.
- Pin or constrain dependencies based on a successfully tested Python 3.14 environment; do not guess package pins.
- Reconcile README, in-app methodology, complexity claims, test count, setup instructions, and actual implemented features.
- Run a fresh-environment test and an end-to-end manual smoke test on the user's Windows machine.

**Acceptance gate:**
- All unit and integration tests pass locally and in CI.
- The full workflow completes on the target machine using the actual saved data and embeddings.
- All charts and result cards are sourced from real artifacts or experiment records.
- README and methodology agree with the code.
- A reviewed pull request is merged only after the user has verified the local run.

---

## 4. Website design direction

The project should remain a **professional research and experimentation dashboard**, not a generic marketing website. The current dark navy and cyan identity is retained and refined.

### Visual system

- Background: deep navy `#0B1220`.
- Main panels: `#111C33`.
- Primary accent: cyan `#22D3EE`.
- Main text: `#E5E7EB`; secondary text: muted slate/grey.
- Use restrained borders, consistent spacing, moderate corner radii, and clear typography.
- Use status colours only for real states: green for verified success, amber for warnings, red for failures.
- Avoid decorative gradients, excessive animation, large hero banners, and dense tables with tiny text.
- Keep charts in the dark theme with clear axes, units, legends, and explanatory captions.

### Information architecture

1. **Dashboard** — current dataset and embedding identity, validated stage status, latest experiment, open issues, and the next available action.
2. **Dataset** — source, sampling settings, class balance, cleaning report, fingerprint, validation, and sample records.
3. **Embeddings** — model/device/batch configuration, generation/cache state, dimension, norms, storage, and nearest-neighbour examples.
4. **kNN Graph** — k selector, graph construction, connectivity/degree/weight metrics, cache identity, and a bounded neighbourhood explorer.
5. **MST Lab** — choose a graph, run Prim/Kruskal/Borůvka, compare timings and weights, inspect validation checks, and explore a small graph subview.
6. **MST Clustering** — choose a validated forest, configure target and minimum cluster size, inspect cut edges, cluster sizes, topic composition for evaluation, and sample documents.
7. **Baseline Lab** — run K-means and HDBSCAN, display parameters, cluster counts, HDBSCAN noise coverage, and clearly differentiated evaluation protocols.
8. **Experiment Center** — configure repeatable experiments, show resource estimates before dense-graph runs, track progress, and preserve a complete run manifest.
9. **Results & Reports** — filter experiment history, compare algorithms and baselines fairly, inspect run-to-run variability, and export CSV/JSON/report files.
10. **Methodology & Settings** — equations, complexity assumptions, algorithm explanations, metric definitions, hardware/software metadata, and reproducibility controls.

### Interaction requirements

- Every action must state its prerequisite and the artifact it will produce.
- Disable actions when prerequisites are missing or stale; explain how to resolve the blocker.
- Show progress for expensive operations and preserve the last valid result if a new run fails.
- Ask for confirmation before overwriting artifacts or deleting experiment history.
- Let users inspect a small neighbourhood/subgraph rather than rendering thousands of edges at once.
- Never display a green “complete” state solely because a button was clicked. Completion requires validated output.

---

## 5. End-to-end acceptance checklist

The project is ready for final release only when all of the following are true:

- [ ] Clean setup succeeds with the documented Python version and dependencies.
- [ ] Dataset integrity and fingerprint checks pass.
- [ ] Embedding cache validation and reload pass.
- [ ] kNN graph construction is deterministic and validated for all configured k values.
- [ ] Prim, Kruskal, and Borůvka pass independent MST/MSF validation on the same graph.
- [ ] Edge cutting meets minimum-size constraints and reports target reachability correctly.
- [ ] K-means and HDBSCAN are documented and evaluated according to their actual methods.
- [ ] ARI, NMI, silhouette, noise coverage, and exclusions are reported correctly.
- [ ] Experiment C uses matching, deterministic sample indices for complete and sparse graphs.
- [ ] Runtime and memory measurements are accurately labelled.
- [ ] Experiment records contain complete reproducibility metadata.
- [ ] Dataset/embedding/graph changes invalidate every affected downstream artifact.
- [ ] Every website page has correct prerequisites, status, error, and empty states.
- [ ] The dashboard, charts, and exports use actual measured data only.
- [ ] All tests pass locally and through GitHub Actions.
- [ ] README, methodology, and implementation agree.
- [ ] The user verifies the final workflow on their Windows/Python 3.14 setup.

## 6. Execution order

Work in this order: **A → B → C → D → E → F → G → H → I**.

Do not start broad visual redesign before core data, algorithm, clustering, and metric correctness are stable. Small UI fixes needed to test a phase are allowed, but the comprehensive UX pass belongs in Phase I.

At the end of each phase, provide:
- files changed;
- key implementation decisions;
- tests added/changed;
- exact test command and actual result;
- remaining known limitations;
- local run instructions;
- commit/branch reference.

The user should only need to pull the branch, install documented dependencies if needed, run the commands, and report any local-only result. The work should remain phase-by-phase, with no unverified completion claims.
