"""Light text cleaning suited to sentence-transformer models."""
import html

import pandas as pd

from src import config

PREPROCESSING_STEPS = [
    "HTML entities decoded (for example &amp; becomes &).",
    "Backslashes replaced by spaces (AG News uses them as line-break markers).",
    "Whitespace normalised: runs of spaces/newlines become one space, ends trimmed.",
    "Records with missing/empty text or an invalid label removed before sampling.",
    "Case, punctuation and stop words left unchanged (sentence-transformers need them).",
]


def clean_text(text) -> str:
    """Return cleaned text; missing values become an empty string."""
    if text is None or pd.isna(text):
        return ""
    text = html.unescape(str(text))
    text = text.replace("\\", " ")
    return " ".join(text.split())


def preprocess_dataframe(df: pd.DataFrame):
    """Clean the text column and drop invalid records.

    Returns (cleaned_dataframe, removed_counts).
    """
    out = df.copy()
    out["text"] = out["text"].map(clean_text)
    empty = out["text"].str.len() == 0
    bad_label = ~out["label"].isin(list(config.LABEL_NAMES))
    keep = ~empty & ~bad_label
    removed = {
        "empty_text": int(empty.sum()),
        "invalid_label": int((bad_label & ~empty).sum()),
    }
    return out[keep].reset_index(drop=True), removed
