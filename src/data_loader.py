"""Load AG News, build the balanced subset, validate it, save/load it."""
import hashlib
import json
from datetime import datetime, timezone

import pandas as pd

from src import config
from src.preprocessing import PREPROCESSING_STEPS, preprocess_dataframe


def load_raw_ag_news() -> pd.DataFrame:
    """Download (or read from the local Hugging Face cache) the AG News train split."""
    from datasets import load_dataset

    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)
    df = dataset.to_pandas()[["text", "label"]]
    df.insert(0, "doc_id", [f"agnews_{config.DATASET_SPLIT}_{i:06d}" for i in range(len(df))])
    return df


def build_balanced_subset(raw_df, per_class=config.SAMPLES_PER_CLASS, seed=config.RANDOM_SEED):
    """Clean the data, then draw `per_class` random documents from each class."""
    clean_df, removed = preprocess_dataframe(raw_df)

    parts = []
    for label in sorted(config.LABEL_NAMES):
        group = clean_df[clean_df["label"] == label]
        if len(group) < per_class:
            raise ValueError(
                f"Class {config.LABEL_NAMES[label]} has only {len(group)} valid "
                f"documents, but {per_class} are needed."
            )
        parts.append(group.sample(n=per_class, random_state=seed))

    subset = pd.concat(parts).sort_values("doc_id").reset_index(drop=True)
    subset["label"] = subset["label"].astype(int)
    subset["label_name"] = subset["label"].map(config.LABEL_NAMES)
    subset = subset[["doc_id", "text", "label", "label_name"]]

    report = {
        "raw_rows": int(len(raw_df)),
        "removed_empty_text": removed["empty_text"],
        "removed_invalid_label": removed["invalid_label"],
    }
    return subset, report


def subset_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of ids, labels and texts: identical subsets give identical fingerprints."""
    digest = hashlib.sha256()
    for row in df.itertuples(index=False):
        digest.update(f"{row.doc_id}|{row.label}|{row.text}\n".encode("utf-8"))
    return digest.hexdigest()


def validate_subset(df: pd.DataFrame, per_class=config.SAMPLES_PER_CLASS):
    """Return a list of (check name, passed, detail)."""
    checks = []
    expected_total = per_class * len(config.LABEL_NAMES)
    checks.append(("Total document count", len(df) == expected_total,
                   f"{len(df)} found, {expected_total} expected"))

    counts = df["label"].value_counts().to_dict()
    balanced = all(counts.get(label, 0) == per_class for label in config.LABEL_NAMES)
    checks.append(("Every class has the expected count", balanced,
                   ", ".join(f"{name}: {counts.get(label, 0)}"
                             for label, name in config.LABEL_NAMES.items())))

    empty = int((df["text"].isna() | (df["text"].astype(str).str.strip() == "")).sum())
    checks.append(("No missing or empty text", empty == 0, f"{empty} empty"))

    valid_labels = bool(df["label"].isin(list(config.LABEL_NAMES)).all())
    checks.append(("All labels are valid (0 to 3)", valid_labels, ""))

    names_match = bool((df["label"].map(config.LABEL_NAMES) == df["label_name"]).all())
    checks.append(("Label ids and label names agree", names_match, ""))

    unique_ids = bool(df["doc_id"].is_unique)
    checks.append(("Document IDs are unique", unique_ids, ""))
    return checks


def save_subset(df: pd.DataFrame, meta: dict) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.SUBSET_CSV, index=False, encoding="utf-8")
    with open(config.SUBSET_META, "w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)


def load_saved_subset():
    """Return (dataframe, metadata) if a subset was saved earlier, else None."""
    if not (config.SUBSET_CSV.exists() and config.SUBSET_META.exists()):
        return None
    df = pd.read_csv(config.SUBSET_CSV, encoding="utf-8", keep_default_na=False)
    with open(config.SUBSET_META, encoding="utf-8") as file:
        meta = json.load(file)
    return df, meta


def create_and_save_subset(per_class=config.SAMPLES_PER_CLASS, seed=config.RANDOM_SEED):
    """Full Phase 2 pipeline: load -> clean -> sample -> validate -> save."""
    import datasets

    raw = load_raw_ag_news()
    subset, report = build_balanced_subset(raw, per_class, seed)

    failed = [name for name, passed, _ in validate_subset(subset, per_class) if not passed]
    if failed:
        raise ValueError("Validation failed: " + ", ".join(failed))

    class_counts = subset["label_name"].value_counts()
    meta = {
        "dataset": config.DATASET_NAME,
        "split": config.DATASET_SPLIT,
        "random_seed": seed,
        "samples_per_class": per_class,
        "total_documents": int(len(subset)),
        "class_counts": {name: int(class_counts[name]) for name in config.LABEL_NAMES.values()},
        **report,
        "preprocessing_steps": PREPROCESSING_STEPS,
        "fingerprint_sha256": subset_fingerprint(subset),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets_library_version": datasets.__version__,
    }
    save_subset(subset, meta)
    return subset, meta