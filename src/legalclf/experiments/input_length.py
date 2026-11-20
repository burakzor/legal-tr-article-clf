"""Experiment 2: how much of an article does logistic regression need to recognize the legislation type?

Usage:
    python -m legalclf.experiments.input_length --config configs/experiments/input_length_logreg.yaml

For each input length k: truncate train/dev/test to the first k words, fit TF-IDF on train,
grid-search lr x weight_decay on dev, then evaluate the best model once on test.
Outputs in `out_dir`: results.csv (all test metrics per k), per_class_f1.csv, search.csv,
history.csv, input_length.png, per_class_vs_length.png, config.json, and for every k a folder
k_<k>/ with the full evaluation report (metrics.json, per_class.csv with TP/FP/FN/TN,
confusion matrix, ROC / PR curves, score histograms, threshold trade-off, training curve).
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
import torch
import yaml

from legalclf import plots
from legalclf.data import encode_labels, load_splits
from legalclf.evaluation import write_report
from legalclf.features import make_tfidf, truncate_words
from legalclf.models import logistic_regression as logreg
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "torch"]


def k_name(k: int | None) -> str:
    return "full" if k is None else str(k)


def run_length(k: int | None, splits: dict, y: dict, cfg: dict, device: str, out_dir: Path) -> dict:
    """Search on dev for one input length; return the best model's test results and the search log."""
    texts = {s: truncate_words(df["text"], k) for s, df in splits.items()}
    vec = make_tfidf(cfg["tfidf"])
    X = {"train": vec.fit_transform(texts["train"])}
    X["dev"], X["test"] = vec.transform(texts["dev"]), vec.transform(texts["test"])

    search, histories, best = [], [], None
    for lr in cfg["search"]["lr"]:
        for wd in cfg["search"]["weight_decay"]:
            start = time.time()
            model, history = logreg.fit(X["train"], y["train"], X["dev"], y["dev"], len(LABELS),
                                         lr=float(lr), weight_decay=float(wd), seed=cfg["seed"],
                                         device=device, **cfg["model"])
            dev_f1 = max(h["dev_macro_f1"] for h in history)
            row = {"k": k_name(k), "lr": lr, "weight_decay": wd, "dev_macro_f1": dev_f1,
                   "best_epoch": int(np.argmax([h["dev_macro_f1"] for h in history])) + 1,
                   "seconds": round(time.time() - start, 1)}
            search.append(row)
            histories += [{"k": k_name(k), "lr": lr, "weight_decay": wd, **h} for h in history]
            print(f"  k={k_name(k):>4}  lr={lr:<7} wd={wd:<7} dev macro-F1={dev_f1:.4f}  ({row['seconds']}s)")
            if best is None or dev_f1 > best[0]["dev_macro_f1"]:
                best = (row, model)

    row, model = best
    k_dir = out_dir / f"k_{k_name(k)}"
    proba = logreg.predict_proba(model, X["test"], device)
    test = write_report(k_dir, y["test"], proba, LABELS, splits["test"]["doc_id"], probabilistic=True,
                        bootstrap_n=cfg["bootstrap"]["n"], seed=cfg["seed"],
                        title=f"Logistic regression, {'full articles' if k is None else f'first {k} words'}, test set")
    best_history = pd.DataFrame([h for h in histories if h["lr"] == row["lr"] and h["weight_decay"] == row["weight_decay"]])
    plots.training_curves(best_history, k_dir / "training_curve.png", f"lr={row['lr']}, weight_decay={row['weight_decay']}")
    result = {
        "k": k_name(k),
        "share_train_cut": float((splits["train"]["n_words"] > k).mean()) if k is not None else 0.0,
        "n_features": X["train"].shape[1],
        "lr": row["lr"], "weight_decay": row["weight_decay"], "best_epoch": row["best_epoch"],
        "dev_macro_f1": row["dev_macro_f1"],
        **{f"test_{m}": v for m, v in test.items() if m != "n"},
    }
    f1s = pd.read_csv(k_dir / "per_class.csv")["f1"]
    return {"result": result, "per_class": dict(zip(LABELS, f1s)), "search": search, "history": histories}


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
    ax.set_title("Logistic regression: effect of input length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, path: Path) -> None:
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_splits(root, cfg["data_version"])
    y = {s: encode_labels(df["label"]).to_numpy() for s, df in splits.items()}
    print(f"device={device}  articles: " + ", ".join(f"{s}={len(df):,}" for s, df in splits.items()))

    results, per_class, search, history = [], [], [], []
    for k in cfg["input_lengths"]:
        out = run_length(k, splits, y, cfg, device, out_dir)
        r = out["result"]
        print(f"k={r['k']:>4}  test macro-F1={r['test_macro_f1']:.4f} "
              f"[{r['test_macro_f1_low']:.4f}, {r['test_macro_f1_high']:.4f}]  accuracy={r['test_accuracy']:.4f}  "
              f"balanced acc={r['test_balanced_accuracy']:.4f}  macro ROC-AUC={r['test_macro_roc_auc']:.4f}")
        results.append(r)
        per_class.append({"k": r["k"], **out["per_class"]})
        search += out["search"]
        history += out["history"]

    results = pd.DataFrame(results)
    results.to_csv(out_dir / "results.csv", index=False)
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(out_dir / "per_class_f1.csv", index=False)
    plot_per_class(per_class, out_dir / "per_class_vs_length.png")
    pd.DataFrame(search).to_csv(out_dir / "search.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot(results, out_dir / "input_length.png")
    record = {"config": cfg, "device": device, "created": date.today().isoformat(),
              "packages": {p: version(p) for p in PACKAGES}}
    (out_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
