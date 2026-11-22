"""Experiment 2: how much of an article does Naive Bayes need to recognize the legislation type?

Usage:
    python -m legalclf.experiments.input_length_nb --config configs/experiments/input_length_nb.yaml

Same procedure and outputs as `input_length.py` (logistic regression), so the two runs can be
compared directly. For each input length k: truncate train/dev/test to the first k words, fit
TF-IDF on train, grid-search variant x alpha x fit_prior on dev, then evaluate the best model
once on test.
Outputs in `out_dir`: results.csv (all test metrics per k), per_class_f1.csv, search.csv,
input_length.png, per_class_vs_length.png, config.json, and for every k a folder k_<k>/ with the
full evaluation report (metrics.json, per_class.csv with TP/FP/FN/TN, confusion matrix,
ROC / PR curves, score histograms, threshold trade-off). Naive Bayes is fit in closed form
(no epochs), so there is no history.csv or training curve.
"""
import argparse
import json
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from legalclf.data import encode_labels, load_splits
from legalclf.evaluation import write_report
from legalclf import plots
from legalclf.features import make_tfidf, truncate_words
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn"]


def k_name(k: int | None) -> str:
    return "full" if k is None else str(k)


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


def candidates(search: dict):
    """(params, unfitted model) for every point of the grid. fit_prior only applies to MultinomialNB."""
    for variant in search["variant"]:
        for alpha in search["alpha"]:
            if variant == "multinomial":
                for fit_prior in search["fit_prior"]:
                    yield ({"variant": variant, "alpha": alpha, "fit_prior": fit_prior},
                           MultinomialNB(alpha=float(alpha), fit_prior=bool(fit_prior)))
            elif variant == "complement":
                yield ({"variant": variant, "alpha": alpha, "fit_prior": None},
                       ComplementNB(alpha=float(alpha)))
            else:
                raise ValueError(f"unknown variant: {variant}")


def run_length(k: int | None, splits: dict, y: dict, cfg: dict, out_dir: Path) -> dict:
    """Search on dev for one input length; return the best model's test results and the search log."""
    texts = {s: truncate_words(df["text"], k) for s, df in splits.items()}
    vec = make_tfidf(cfg["tfidf"])
    X = {"train": vec.fit_transform(texts["train"])}
    X["dev"], X["test"] = vec.transform(texts["dev"]), vec.transform(texts["test"])

    search, best = [], None
    for params, model in candidates(cfg["search"]):
        start = time.time()
        model.fit(X["train"], y["train"])
        dev_f1 = f1_score(y["dev"], model.predict(X["dev"]), labels=list(range(len(LABELS))),
                          average="macro", zero_division=0)
        row = {"k": k_name(k), **params, "dev_macro_f1": dev_f1, "seconds": round(time.time() - start, 2)}
        search.append(row)
        if best is None or dev_f1 > best[0]["dev_macro_f1"]:
            best = (row, model)

    row, model = best
    print(f"  k={k_name(k):>4}  best on dev: {row['variant']} alpha={row['alpha']} "
          f"fit_prior={row['fit_prior']}  dev macro-F1={row['dev_macro_f1']:.4f}")
    k_dir = out_dir / f"k_{k_name(k)}"
    # model.classes_ is 0..6 because every class occurs in train, so columns follow LABELS.
    proba = model.predict_proba(X["test"])
    test = write_report(k_dir, y["test"], proba, LABELS, splits["test"]["doc_id"], probabilistic=True,
                        bootstrap_n=cfg["bootstrap"]["n"], seed=cfg["seed"],
                        title=f"Naive Bayes, {'full articles' if k is None else f'first {k} words'}, test set")
    result = {
        "k": k_name(k),
        "share_train_cut": float((splits["train"]["n_words"] > k).mean()) if k is not None else 0.0,
        "n_features": X["train"].shape[1],
        "variant": row["variant"], "alpha": row["alpha"], "fit_prior": row["fit_prior"],
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
    ax.set_title("Naive Bayes: effect of input length")
    ax.grid(alpha=0.3)
    ax.legend()
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
