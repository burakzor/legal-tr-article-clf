"""Metrics shared by all models: macro-F1, accuracy and a document-level bootstrap interval."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    labels = list(range(n_classes))
    return {
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)


def bootstrap_macro_f1(doc_ids: pd.Series, y_true: np.ndarray, y_pred: np.ndarray, n_classes: int,
                       n: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    """95% interval of macro-F1, resampling documents (articles of one document are not independent)."""
    groups = [np.asarray(ix) for ix in pd.Series(np.arange(len(doc_ids))).groupby(doc_ids.values).groups.values()]
    rng = np.random.default_rng(seed)
    labels = list(range(n_classes))
    values = []
    for _ in range(n):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        values.append(f1_score(y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0))
    return float(np.quantile(values, alpha / 2)), float(np.quantile(values, 1 - alpha / 2))
