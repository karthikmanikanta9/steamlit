import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import build_balanced_subset, subset_fingerprint, validate_subset
from src.preprocessing import clean_text, preprocess_dataframe


def make_raw(per_class=20):
    rows, i = [], 0
    for label in config.LABEL_NAMES:
        for _ in range(per_class):
            rows.append({"doc_id": f"d{i:04d}", "text": f"Document {i} on topic {label}", "label": label})
            i += 1
    return pd.DataFrame(rows)


def test_clean_text_normalises_whitespace_and_symbols():
    assert clean_text("  Hello   \n world \t ") == "Hello world"
    assert clean_text("Wall\\Street") == "Wall Street"
    assert clean_text("AT&amp;T") == "AT&T"


def test_clean_text_handles_missing_values():
    assert clean_text(None) == ""
    assert clean_text(np.nan) == ""
    assert clean_text("   ") == ""


def test_preprocess_removes_empty_and_invalid_records():
    df = make_raw(5)
    df.loc[0, "text"] = "   "
    df.loc[1, "text"] = None
    df.loc[2, "label"] = 9
    cleaned, removed = preprocess_dataframe(df)
    assert removed == {"empty_text": 2, "invalid_label": 1}
    assert len(cleaned) == len(df) - 3


def test_subset_is_balanced_and_valid():
    subset, report = build_balanced_subset(make_raw(), per_class=10, seed=1)
    assert len(subset) == 40
    assert subset["label"].value_counts().tolist() == [10, 10, 10, 10]
    assert all(passed for _, passed, _ in validate_subset(subset, per_class=10))
    assert report["raw_rows"] == 80


def test_same_seed_gives_same_subset_and_different_seed_differs():
    raw = make_raw()
    a, _ = build_balanced_subset(raw, per_class=10, seed=7)
    b, _ = build_balanced_subset(raw, per_class=10, seed=7)
    c, _ = build_balanced_subset(raw, per_class=10, seed=8)
    assert subset_fingerprint(a) == subset_fingerprint(b)
    assert subset_fingerprint(a) != subset_fingerprint(c)


def test_labels_stay_aligned_with_text():
    raw = make_raw()
    subset, _ = build_balanced_subset(raw, per_class=10, seed=3)
    merged = subset.merge(raw, on="doc_id", suffixes=("_sub", "_raw"))
    assert len(merged) == len(subset)
    assert (merged["text_sub"] == merged["text_raw"]).all()
    assert (merged["label_sub"] == merged["label_raw"]).all()


def test_too_few_documents_raises_error():
    with pytest.raises(ValueError):
        build_balanced_subset(make_raw(5), per_class=10, seed=1)


def test_validation_detects_problems():
    subset, _ = build_balanced_subset(make_raw(), per_class=10, seed=1)
    broken = subset.copy()
    broken.loc[0, "text"] = ""
    broken.loc[1, "doc_id"] = broken.loc[2, "doc_id"]
    failed = [name for name, passed, _ in validate_subset(broken, per_class=10) if not passed]
    assert "No missing or empty text" in failed
    assert "Document IDs are unique" in failed