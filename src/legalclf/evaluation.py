"""Metrics shared by all models (CS464 Ch. 7 performance metrics, multi-class).

Per class (one-vs-rest): TP, FP, FN, TN, precision, recall (sensitivity, TPR), specificity (TNR),
FPR, F1, F0.5, F2, ROC-AUC, average precision. Overall: accuracy, macro / micro / weighted
precision, recall and F1, balanced accuracy, MCC, Cohen's kappa, top-2 accuracy, log loss,
macro / micro ROC-AUC and average precision, and a document-level bootstrap interval of macro-F1.

`scores` is an (n, n_classes) array: probabilities for LR / NB, decision values for SVM
(set probabilistic=False, then log loss is skipped).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score, fbeta_score, log_loss,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)


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


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes, dtype=int)[y]


def per_class_table(y_true: np.ndarray, y_pred: np.ndarray, scores_: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """One row per class, each class treated as 'positive' against the rest."""
    k = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(k)))   # rows = true, columns = predicted
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(k)), zero_division=0)
    Y = one_hot(y_true, k)
    table = pd.DataFrame({
        "class": labels, "support": support, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "precision": precision,
        "recall": recall,                                         # sensitivity, TPR
        "specificity": tn / np.maximum(tn + fp, 1),               # TNR
        "fpr": fp / np.maximum(fp + tn, 1),
        "f1": f1,
        "f0.5": fbeta_score(y_true, y_pred, beta=0.5, labels=list(range(k)), average=None, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, labels=list(range(k)), average=None, zero_division=0),
        "roc_auc": [roc_auc_score(Y[:, c], scores_[:, c]) if 0 < Y[:, c].sum() < len(Y) else np.nan
                    for c in range(k)],
        "average_precision": [average_precision_score(Y[:, c], scores_[:, c]) if Y[:, c].any() else np.nan
                              for c in range(k)],
    })
    return table


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores_: np.ndarray, table: pd.DataFrame,
                    probabilistic: bool) -> dict:
    k = scores_.shape[1]
    labels = list(range(k))
    tp, fp, fn = table["TP"].sum(), table["FP"].sum(), table["FN"].sum()
    out = {"n": int(len(y_true)), "accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=avg, zero_division=0)
        out.update({f"{avg}_precision": p, f"{avg}_recall": r, f"{avg}_f1": f})
    # micro: pool TP / FP / FN over classes (for single-label data this equals accuracy)
    out.update({"micro_precision": tp / (tp + fp), "micro_recall": tp / (tp + fn),
                "micro_f1": 2 * tp / (2 * tp + fp + fn)})
    out.update({
        "macro_specificity": table["specificity"].mean(),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "top2_accuracy": float(np.mean((np.argsort(-scores_, axis=1)[:, :2] == y_true[:, None]).any(axis=1))),
        "macro_roc_auc": table["roc_auc"].mean(),
        "micro_roc_auc": roc_auc_score(one_hot(y_true, k).ravel(), scores_.ravel()),
        "macro_average_precision": table["average_precision"].mean(),
        "micro_average_precision": average_precision_score(one_hot(y_true, k).ravel(), scores_.ravel()),
    })
    if probabilistic:
        out["log_loss"] = log_loss(y_true, np.clip(scores_, 1e-12, 1), labels=labels)
    return {key: float(v) if not isinstance(v, int) else v for key, v in out.items()}


def write_report(out_dir: Path, y_true: np.ndarray, scores_: np.ndarray, labels: list[str], doc_ids: pd.Series,
                 probabilistic: bool = True, bootstrap_n: int = 1000, seed: int = 42, title: str = "") -> dict:
    """Compute every metric, save tables and plots to out_dir, return the summary dict."""
    from legalclf import plots   # matplotlib only when a report is written

    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = scores_.argmax(axis=1)
    table = per_class_table(y_true, y_pred, scores_, labels)
    summary = summary_metrics(y_true, y_pred, scores_, table, probabilistic)
    summary["macro_f1_low"], summary["macro_f1_high"] = bootstrap_macro_f1(
        doc_ids, y_true, y_pred, len(labels), n=bootstrap_n, seed=seed)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    pd.DataFrame(cm, index=labels, columns=labels).rename_axis(index="true", columns="pred").to_csv(
        out_dir / "confusion_counts.csv")
    table.to_csv(out_dir / "per_class.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    Y = one_hot(y_true, len(labels))
    plots.confusion_matrices(cm, labels, out_dir / "confusion_matrix.png", title)
    plots.per_class_bars(table, out_dir / "per_class_metrics.png", title)
    plots.roc_curves(Y, scores_, labels, out_dir / "roc_curves.png", title)
    plots.pr_curves(Y, scores_, labels, out_dir / "pr_curves.png", title)
    plots.score_histograms(Y, scores_, labels, out_dir / "score_histograms.png", title)
    plots.threshold_tradeoff(Y, scores_, labels, out_dir / "threshold_tradeoff.png", title)
    return summary
