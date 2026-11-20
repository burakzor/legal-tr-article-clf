"""Shared pieces for the classical models: data loading, input truncation, text
normalization, metrics and document-level bootstrap."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from legalclf.preprocessing.build import LABELS, SPLITS
from legalclf.preprocessing.clean import truncate

FULL = "full"  # input length meaning "the whole (already ≤1,000 words) article"


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}


def truncate_texts(texts: pd.Series, max_words: int | str) -> pd.Series:
    """Keep the first `max_words` whitespace-separated words of every text (same word
    definition as `n_words`). `FULL` returns the texts unchanged."""
    if max_words == FULL:
        return texts
    return texts.map(lambda t: truncate(t, int(max_words))[0])


def turkish_lower(text: str) -> str:
    """Lowercase with Turkish dotted/dotless i. Plain str.lower() turns "İ" into
    "i" + combining dot and "I" into "i", which splits one word into several tokens."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def macro_f1(y_true, y_pred) -> float:
    """Macro-F1 over the classes that occur in y_true (all 7 on the full dev/test sets)."""
    labels = [l for l in LABELS if l in set(y_true)]
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return dict(zip(LABELS, scores))


def doc_bootstrap_ci(y_true, y_pred, doc_ids, n_boot: int = 1000, seed: int = 42,
                     level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of macro-F1, resampling documents (articles of one document are
    not independent)."""
    y_true, y_pred, doc_ids = map(np.asarray, (y_true, y_pred, doc_ids))
    groups = list(pd.Series(np.arange(len(doc_ids))).groupby(doc_ids).indices.values())
    rng = np.random.default_rng(seed)
    scores = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in picked])
        scores[b] = macro_f1(y_true[idx], y_pred[idx])
    alpha = (1 - level) / 2
    lo, hi = np.quantile(scores, [alpha, 1 - alpha])
    return float(lo), float(hi)
