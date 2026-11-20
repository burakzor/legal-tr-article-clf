"""Evaluation plots (CS464 Ch. 7): confusion matrix, per-class metrics, ROC and PR curves,
histograms of predicted scores and the sensitivity / specificity trade-off over the threshold c."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve  # noqa: E402

SHORT = {"YONETMELIK": "YÖN", "TEBLIG": "TEB", "KANUN": "KAN", "CUMHURBASKANLIGI_YONETMELIGI": "CBY",
         "TUZUK": "TÜZ", "KANUN_HUKMUNDE_KARARNAME": "KHK", "CUMHURBASKANLIGI_KARARNAMESI": "CBK"}


def short(labels: list[str]) -> list[str]:
    return [SHORT.get(label, label) for label in labels]


def _save(fig, path: Path, title: str) -> None:
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _grid(n: int):
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return fig, axes.ravel()


def confusion_matrices(cm: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    """Counts and row-normalized matrix (diagonal of the normalized one = recall of each class)."""
    names = short(labels)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, mat, fmt, name in [(axes[0], cm, "d", "counts"), (axes[1], norm, ".2f", "row-normalized (recall)")]:
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), names)
        ax.set_yticks(range(len(labels)), names)
        ax.set_xlabel("predicted class")
        ax.set_ylabel("true class")
        ax.set_title(f"Confusion matrix, {name}")
    fig.colorbar(im, ax=axes, fraction=0.025)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def per_class_bars(table: pd.DataFrame, path: Path, title: str = "") -> None:
    metrics = ["precision", "recall", "specificity", "f1"]
    x = np.arange(len(table))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, table[m], width, label=m)
    ax.set_xticks(x, [f"{n}\n(n={s})" for n, s in zip(short(list(table["class"])), table["support"])])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Per-class metrics (one-vs-rest)")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path, title)


def roc_curves(Y: np.ndarray, S: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    """One-vs-rest ROC per class, plus micro (pooled) and macro (mean TPR) averages."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    grid = np.linspace(0, 1, 501)
    mean_tpr = np.zeros_like(grid)
    for c, name in enumerate(short(labels)):
        fpr, tpr, _ = roc_curve(Y[:, c], S[:, c])
        mean_tpr += np.interp(grid, fpr, tpr) / len(labels)
        ax.plot(fpr, tpr, lw=1, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    fpr, tpr, _ = roc_curve(Y.ravel(), S.ravel())
    ax.plot(fpr, tpr, "k--", lw=2, label=f"micro avg (AUC={auc(fpr, tpr):.3f})")
    ax.plot(grid, mean_tpr, "k:", lw=2, label=f"macro avg (AUC={auc(grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(fontsize=7, loc="lower right")
    _save(fig, path, title)


def pr_curves(Y: np.ndarray, S: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    """Precision-recall per class; a random classifier's precision equals the class share (in brackets)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c, name in enumerate(short(labels)):
        precision, recall, _ = precision_recall_curve(Y[:, c], S[:, c])
        ap = average_precision_score(Y[:, c], S[:, c])
        ax.plot(recall, precision, lw=1, label=f"{name} (AP={ap:.3f}, base={Y[:, c].mean():.3f})")
    precision, recall, _ = precision_recall_curve(Y.ravel(), S.ravel())
    ax.plot(recall, precision, "k--", lw=2,
            label=f"micro avg (AP={average_precision_score(Y.ravel(), S.ravel()):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=7, loc="lower left")
    _save(fig, path, title)


def score_histograms(Y: np.ndarray, S: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    """For each class: predicted score of articles that belong to it vs. articles that do not."""
    fig, axes = _grid(len(labels))
    bins = np.linspace(S.min(), S.max(), 41)
    for c, (ax, name) in enumerate(zip(axes, short(labels))):
        ax.hist(S[Y[:, c] == 0, c], bins=bins, density=True, alpha=0.6, label="other classes")
        ax.hist(S[Y[:, c] == 1, c], bins=bins, density=True, alpha=0.6, label=name)
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel(f"predicted score for {name}")
        ax.legend(fontsize=7)
    _save(fig, path, title or "Histograms of predicted scores")


def threshold_tradeoff(Y: np.ndarray, S: np.ndarray, labels: list[str], path: Path, title: str = "") -> None:
    """Sensitivity and specificity as the threshold c moves; the dashed line is where they cross."""
    fig, axes = _grid(len(labels))
    for c, (ax, name) in enumerate(zip(axes, short(labels))):
        fpr, tpr, thr = roc_curve(Y[:, c], S[:, c])
        keep = np.isfinite(thr)
        thr, sens, spec = thr[keep], tpr[keep], 1 - fpr[keep]
        ax.plot(thr, sens, label="sensitivity")
        ax.plot(thr, spec, label="specificity")
        cross = thr[np.argmin(np.abs(sens - spec))]
        ax.axvline(cross, color="grey", linestyle="--", lw=0.8)
        ax.text(cross, 0.05, f" c={cross:.3f}", fontsize=7)
        ax.set_title(name)
        ax.set_xlabel("threshold c")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
    _save(fig, path, title or "Sensitivity / specificity trade-off")


def training_curves(history: pd.DataFrame, path: Path, title: str = "") -> None:
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["epoch"], history["train_loss"], marker="o", color="tab:blue", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["dev_macro_f1"], marker="s", color="tab:orange", label="dev macro-F1")
    ax2.set_ylabel("dev macro-F1", color="tab:orange")
    ax1.set_title("Training curve of the selected model")
    _save(fig, path, title)
