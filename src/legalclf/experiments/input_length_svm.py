"""Experiment 2: how much of an article does a linear SVM need to recognize the legislation type?

Usage:
    python -m legalclf.experiments.input_length_svm --config configs/experiments/input_length_svm.yaml

Same procedure and outputs as `input_length.py` (logistic regression) and `input_length_nb.py`
(Naive Bayes), so the three runs can be compared directly. The model and its search grid are the
ones of `svm.py` (models/svm.py, configs/experiments/svm.yaml), so k = full reproduces that result.
For each input length k: truncate train/dev/test to the first k words, fit TF-IDF on train,
grid-search C x class_weight on dev, then evaluate the best model once on test.
Outputs in `out_dir`: results.csv (all test metrics per k), per_class_f1.csv, search.csv,
input_length.png, per_class_vs_length.png, config.json, and for every k a folder k_<k>/ with the
full evaluation report (metrics.json, per_class.csv with TP/FP/FN/TN, confusion matrix,
ROC / PR curves, score histograms, threshold trade-off). The SVM has no probabilities: reports
use decision values (probabilistic=False, no log loss), and there are no epochs, so no history.csv.
"""
import argparse
import json
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from legalclf import plots
from legalclf.data import encode_labels, load_splits
from legalclf.evaluation import scores, write_report
from legalclf.features import make_tfidf, truncate_words
from legalclf.models import svm
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy"]


def k_name(k: int | None) -> str:
    return "full" if k is None else str(k)


def run_length(k: int | None, splits: dict, y: dict, cfg: dict, out_dir: Path) -> dict:
    """Search on dev for one input length; return the best model's test results and the search log."""
    texts = {s: truncate_words(df["text"], k) for s, df in splits.items()}
    vec = make_tfidf(cfg["tfidf"])
    X = {"train": vec.fit_transform(texts["train"])}
    X["dev"], X["test"] = vec.transform(texts["dev"]), vec.transform(texts["test"])

    search, best = [], None
    # Same loop order as svm.py, so ties are broken the same way.
    for cw in cfg["search"]["class_weight"]:
        for C in cfg["search"]["C"]:
            start = time.time()
            model = svm.fit(X["train"], y["train"], C=float(C), class_weight=cw, seed=cfg["seed"], **cfg["model"])
            dev = scores(y["dev"], svm.predict(model, X["dev"], len(LABELS)), len(LABELS))
            row = {"k": k_name(k), "C": C, "class_weight": cw, "dev_macro_f1": dev["macro_f1"],
                   "dev_accuracy": dev["accuracy"], "n_iter": int(np.max(model.n_iter_)),
                   "seconds": round(time.time() - start, 1)}
            search.append(row)
            if best is None or row["dev_macro_f1"] > best[0]["dev_macro_f1"]:
                best = (row, model)

    row, model = best
    print(f"  k={k_name(k):>4}  best on dev: C={row['C']} class_weight={row['class_weight']}  "
          f"dev macro-F1={row['dev_macro_f1']:.4f}")
    k_dir = out_dir / f"k_{k_name(k)}"
    decision = svm.decision_scores(model, X["test"], len(LABELS))
    test = write_report(k_dir, y["test"], decision, LABELS, splits["test"]["doc_id"], probabilistic=False,
                        bootstrap_n=cfg["bootstrap"]["n"], seed=cfg["seed"],
                        title=f"Linear SVM, {'full articles' if k is None else f'first {k} words'}, test set")
    result = {
        "k": k_name(k),
        "share_train_cut": float((splits["train"]["n_words"] > k).mean()) if k is not None else 0.0,
        "n_features": X["train"].shape[1],
        "C": row["C"], "class_weight": row["class_weight"], "n_iter": row["n_iter"],
        "dev_macro_f1": row["dev_macro_f1"],
        **{f"test_{m}": v for m, v in test.items() if m != "n"},
    }
    f1s = pd.read_csv(k_dir / "per_class.csv")["f1"]
    return {"result": result, "per_class": dict(zip(LABELS, f1s)), "search": search}


def plot(results: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(6, 4))
    yerr = [results["test_macro_f1"] - results["test_macro_f1_low"],
            results["test_macro_f1_high"] - results["test_macro_f1"]]
    ax.errorbar(x, results["test_macro_f1"], yerr=yerr, marker="o", capsize=3, label="test macro-F1 (95% doc bootstrap)")
    ax.plot(x, results["dev_macro_f1"], marker="s", linestyle="--", label="dev macro-F1 (best of search)")
    ax.plot(x, results["test_accuracy"], marker="^", linestyle=":", label="test accuracy (= micro-F1)")
    ax.plot(x, results["test_balanced_accuracy"], marker="v", linestyle=":", label="test balanced accuracy")
    ax.set_xticks(x, results["k"])
    ax.set_xlabel("Input length (first k words of the article)")
    ax.set_ylabel("score")
    ax.set_title("Linear SVM: effect of input length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, path: Path) -> None:
    """Same figure as in input_length.py (copied so this runner does not need torch)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(per_class))
    for label, name in zip(LABELS, plots.short(LABELS)):
        ax.plot(x, per_class[label], marker="o", label=name)
    ax.set_xticks(x, per_class["k"])
    ax.set_xlabel("Input length (first k words of the article)")
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class F1 by input length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = find_repo_root(Path.cwd())
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(root, cfg["data_version"])
    y = {s: encode_labels(df["label"]).to_numpy() for s, df in splits.items()}
    print("articles: " + ", ".join(f"{s}={len(df):,}" for s, df in splits.items()))

    results, per_class, search = [], [], []
    for k in cfg["input_lengths"]:
        out = run_length(k, splits, y, cfg, out_dir)
        r = out["result"]
        print(f"k={r['k']:>4}  test macro-F1={r['test_macro_f1']:.4f} "
              f"[{r['test_macro_f1_low']:.4f}, {r['test_macro_f1_high']:.4f}]  accuracy={r['test_accuracy']:.4f}  "
              f"balanced acc={r['test_balanced_accuracy']:.4f}  macro ROC-AUC={r['test_macro_roc_auc']:.4f}")
        results.append(r)
        per_class.append({"k": r["k"], **out["per_class"]})
        search += out["search"]

    results = pd.DataFrame(results)
    results.to_csv(out_dir / "results.csv", index=False)
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(out_dir / "per_class_f1.csv", index=False)
    plot_per_class(per_class, out_dir / "per_class_vs_length.png")
    pd.DataFrame(search).to_csv(out_dir / "search.csv", index=False)
    plot(results, out_dir / "input_length.png")
    record = {"config": cfg, "created": date.today().isoformat(),
              "packages": {p: version(p) for p in PACKAGES}}
    (out_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
