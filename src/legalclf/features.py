"""Text features shared by the classical models."""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def truncate_words(texts: pd.Series, k: int | None) -> pd.Series:
    """Keep the first k words of each text; None keeps the whole text."""
    if k is None:
        return texts
    return texts.str.split().str[:k].str.join(" ")


def make_tfidf(cfg: dict) -> TfidfVectorizer:
    """Build the vectorizer from the `tfidf` config section. Fit it on train only."""
    return TfidfVectorizer(
        ngram_range=tuple(cfg["ngram_range"]),
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
    )
