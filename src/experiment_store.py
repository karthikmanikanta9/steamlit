"""Save and load experiment records (Phase 7).

Records are stored as JSON Lines (one JSON object per line) in
`results/experiments/experiment_log.jsonl`.  New records are appended;
nothing is ever overwritten.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_experiment(record: dict) -> str:
    """Append one experiment record and return its generated ID."""
    config.EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_id = datetime.now(timezone.utc).strftime("exp_%Y%m%d_%H%M%S_%f")
    full_record = {
        "exp_id": exp_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **record,
    }
    with open(config.EXPERIMENT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(full_record, ensure_ascii=False, default=str) + "\n")
    return exp_id


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_experiments() -> list[dict]:
    """Return all saved experiment records as a list of dicts."""
    path = config.EXPERIMENT_LOG
    if not path.exists():
        return []
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass   # skip malformed lines silently
    return records


def experiments_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Flatten nested experiment records into a display-friendly DataFrame."""
    if not records:
        return pd.DataFrame()

    def _ms(seconds):
        return round(seconds * 1000, 2) if seconds not in (None, "") else None

    def _mb(byt):
        return round(byt / 1_000_000, 3) if byt not in (None, "") else None

    rows = []
    for r in records:
        row = {
            "ID": r.get("exp_id", ""),
            "Time (UTC)": r.get("timestamp_utc", "")[:19],
            "Dataset size": r.get("dataset_size", ""),
            "k": r.get("k", ""),
            "Algorithm": r.get("algorithm", ""),
            "MST time (ms)": _ms(r.get("mst_runtime_seconds")),
            "MST mem (MB)": _mb(r.get("mst_peak_memory_bytes")),
            "MST weight": r.get("mst_total_weight", ""),
            "# clusters": r.get("n_clusters", ""),
            "ARI": r.get("ari", ""),
            "NMI": r.get("nmi", ""),
            "Silhouette": r.get("silhouette", ""),
            "K-means time (ms)": _ms(r.get("kmeans_runtime_seconds")),
            "HDBSCAN time (ms)": _ms(r.get("hdbscan_runtime_seconds")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def clear_experiments() -> None:
    """Delete the experiment log (use with care)."""
    if config.EXPERIMENT_LOG.exists():
        config.EXPERIMENT_LOG.unlink()
