"""Central settings for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUBSET_CSV = PROCESSED_DIR / "ag_news_subset.csv"
SUBSET_META = PROCESSED_DIR / "ag_news_subset_meta.json"

DATASET_NAME = "fancyzhx/ag_news"
DATASET_SPLIT = "train"
LABEL_NAMES = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

SAMPLES_PER_CLASS = 750   # 4 classes x 750 = 3,000 documents
RANDOM_SEED = 42

# ---------- Phase 3: embeddings ----------
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EMBEDDINGS_NPZ = EMBEDDINGS_DIR / "embeddings.npz"
EMBEDDINGS_META = EMBEDDINGS_DIR / "embeddings_meta.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
EMBEDDING_BATCH_SIZE = 64

# ---------- Phase 4 and 5: kNN graph and MST ----------
GRAPHS_DIR = PROJECT_ROOT / "data" / "graphs"
K_VALUES = (5, 10, 15, 20)
KNN_CHUNK_SIZE = 500          # rows processed at a time during the neighbour search
WEIGHT_TOLERANCE = 1e-9       # relative tolerance when comparing total MST weights
TIMING_REPEATS = 5            # timed runs per algorithm (the median is reported)

# ---------- Phase 6: Clustering ----------
DEFAULT_N_CLUSTERS = 4         # target cluster count (matches the 4 AG News classes)
DEFAULT_MIN_CLUSTER_SIZE = 10  # minimum documents per resulting component

# ---------- Phase 6/7: Baselines ----------
KMEANS_SEED = 42
KMEANS_N_INIT = 10
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 5

# ---------- Phase 7: Experiment store ----------
RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EXPERIMENT_LOG = EXPERIMENTS_DIR / "experiment_log.jsonl"
