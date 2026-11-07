# Website Design Specification
## MST-Based Clustering of Text Embeddings

This specification defines the target experience for the existing Streamlit application in `karthikmanikanta9/steamlit`. It is a research dashboard for running and inspecting a real algorithmic pipeline. It must not become a static mock dashboard.

## 1. Design principles

1. Research-first: correctness, provenance, and interpretation take priority over decoration.
2. Workflow-oriented: users should always know what is ready, what is missing, and what action comes next.
3. Evidence-based: every metric and chart must come from a validated artifact or a real experiment record.
4. Explainable: show formulas, parameter values, units, algorithm names, and limitations beside results.
5. Low-resource aware: support CPU-first execution and warn before memory-intensive dense-graph experiments.
6. Consistent: use shared components and terminology across all pages.

## 2. Visual language

| Element | Specification |
|---|---|
| App background | `#0B1220` deep navy |
| Main panel background | `#111C33` |
| Accent | `#22D3EE` cyan |
| Main text | `#E5E7EB` |
| Secondary text | muted slate/grey |
| Success | restrained green |
| Warning | amber |
| Failure | red |
| Borders | subtle low-contrast slate |
| Cards | moderate radius, consistent padding, no excessive shadows |
| Typography | clear hierarchy, readable body text, restrained heading sizes |
| Charts | dark Plotly theme; descriptive axis titles, units, legends and captions |

Keep the existing dark navy/cyan theme. Do not add a second competing design system. Use colour only when it communicates a real status or distinguishes plotted series.

## 3. Global application shell

### Sidebar
- Project mark/title: “MST Text Clustering”.
- Subtitle: “Design & Analysis of Algorithms”.
- Ordered navigation for the workflow.
- Compact “Pipeline readiness” indicators for dataset, embeddings, graph(s), MST, clustering, baselines, and saved experiments.
- A clear link to Methodology and Settings.
- Avoid duplicating a long explanation of every page in the sidebar.

### Main content
Each page should use the same sequence:
1. Page title and one-sentence purpose.
2. Prerequisite/readiness state.
3. Configuration controls.
4. Primary action.
5. Progress/error/success feedback.
6. Results and validation.
7. Interpretation and limitations.
8. Export or inspect details where applicable.

### Shared status language
- **Not created** — no artifact exists.
- **Ready** — an artifact exists and matches its upstream fingerprint.
- **Stale** — an artifact exists but was produced from different input/configuration.
- **Running** — operation is in progress.
- **Validated** — output passed the relevant correctness checks.
- **Failed** — execution or validation failed.

Do not use “complete” for a stage that merely has a file or session-state entry.

## 4. Pages and expected content

### 4.1 Dashboard
- Research question and short pipeline illustration.
- Stage cards showing artifact identity, status and last updated time.
- Dataset row count/class balance and embedding model/dimension.
- Graph k values built and their connected-component counts.
- MST algorithm runs and validation state.
- Latest experiment summary only when a real log record exists.
- A single recommended next action based on missing prerequisites.
- “Known issues” or warnings when artifacts are stale or experiment comparisons are incomplete.

### 4.2 Dataset
- Dataset source and split.
- Sampling seed and samples per class.
- Class distribution chart and table.
- Cleaning counts and validation checklist.
- Dataset fingerprint and creation timestamp.
- Small preview table of articles.
- Clear warning that class labels are not input to the clustering pipeline.

### 4.3 Embeddings
- Model, device, batch size and sequence length.
- Dataset fingerprint used to create the embeddings.
- Cache state and saved artifact size.
- Document count, dimension, dtype, norm validation and generation time.
- Similar-document explorer with cosine similarity and cosine distance.
- Regeneration confirmation when a valid cache exists.

### 4.4 kNN Graph
- k selector and “build selected” / “build configured values” actions.
- Search time, edge-build time, retained edge count, density, average/min/max degree, components, and edge-weight summary.
- Explicit connected graph vs. minimum spanning forest explanation.
- Cache fingerprint and validation status.
- Bounded neighbourhood/subgraph visualization; never attempt to draw the full 3,000-node graph by default.
- Explain exact neighbour search complexity separately from sparse retained edge count.

### 4.5 MST Lab
- Select a validated graph and show V, E, and C before running.
- Run one or all three algorithms on the same immutable graph.
- Show edge count, total weight, median runtime, run spread, and clearly labelled memory measurement.
- Validation table: valid unique IDs, expected V-C edges, acyclic, spanning components, recomputed weight, SciPy reference.
- Explain ties and whether exact edge-set agreement is expected under the implemented tie policy.
- Step-by-step visualization only on a small induced subgraph.

### 4.6 MST Clustering
- Require a validated MST/MSF tied to the selected graph.
- Controls for target cluster count and minimum cluster size.
- Show initial forest components, cuts made, final cluster count, exact target status, and every final cluster size.
- List cut edges and skipped candidates with reasons.
- Topic composition chart explicitly labelled “evaluation only”.
- Sample documents selected deterministically or with a visible sampling seed.
- Show why a target is impossible (e.g. initial forest already has more components than target).

### 4.7 Baseline Lab
- Independent baseline path: embeddings → K-means or HDBSCAN.
- Show exact parameter values and backend/library version.
- K-means must be labelled Euclidean unless a spherical/cosine implementation is explicitly selected and tested.
- HDBSCAN must show cluster count, noise count, coverage fraction and backend.
- Show metrics for all-points (noise as its own label) and clustered-only evaluation where applicable.
- Side-by-side comparison must include coverage and state the scored sample count for each row.

### 4.8 Experiment Center
- Experiment configuration with a clear summary before starting.
- Same deterministic document indices for complete and sparse graph comparisons.
- Sample sizes, k, algorithm, repeat count and random seed.
- Estimated complete-graph edge count and a memory warning/confirmation.
- Progress bar with current step and phase.
- Save the full experiment manifest and measured outputs.
- Distinguish graph construction, neighbour search, MST runtime, graph memory and algorithm working memory.

### 4.9 Results & Reports
- Filter by dataset fingerprint, sample size, k, algorithm, baseline settings, and date.
- Tables and charts for runtime, memory, total weight, cluster count, ARI, NMI, silhouette and HDBSCAN coverage.
- Display repeat counts and variability; do not imply a winner based on one measurement.
- Show the actual latest record after filters are applied, not an unrelated global latest record.
- Export filtered CSV, raw JSONL/JSON manifest, and a readable report.
- Explain empty/missing measurements rather than converting them to zero.

### 4.10 Methodology & Settings
- Research question and pipeline.
- Cosine distance formula and normalized-vector assumption.
- kNN symmetrization rule.
- Prim, Kruskal and Borůvka descriptions and complexities under the actual graph representation.
- MST/MSF validation criteria.
- Edge-cut heuristic and its limitations.
- Baseline distinction and evaluation protocol.
- Exact-search and memory limitations.
- Software versions, hardware, seeds, and artifact lineage.
- No discrepancy between displayed methodology and executed code.

## 5. Interaction and reliability

- Every primary action has disabled/prerequisite states.
- All long operations show progress and preserve the last valid result if the next run fails.
- Confirm destructive actions and cache regeneration.
- Surface actionable exceptions; do not silently continue with stale or malformed artifacts.
- Status must be based on validated lineage keys.
- Changing dataset → invalidates embeddings, graphs, MSTs, clusters, baselines and dependent metrics.
- Changing embeddings → invalidates graphs and every downstream result.
- Changing a graph for one k → invalidates only that k's MST and clustering results.
- Changing evaluation settings → recompute metrics without rerunning the graph/MST where possible.
- Use session state for interaction, not as the only durable storage for expensive computations.

## 6. Performance and accessibility

- CPU-first defaults.
- Never calculate or display a complete graph for all 3,000 records without an explicit resource warning and confirmation.
- Use compact NumPy arrays and chunked calculations.
- Keep visualized graph subviews bounded and expose the node/edge count shown.
- Use readable contrast, meaningful control labels, chart captions, and keyboard-operable standard Streamlit controls.
- Use tables for exact values and charts for trends; do not use charts as the only way to access critical numbers.

## 7. Definition of a polished release

The website is ready when all pages follow the common layout, every stage reflects validated artifact state, all charts are backed by real experiments, error and empty states are complete, methodology matches the implementation, and the end-to-end workflow passes on the target Windows/Python 3.14 environment.
