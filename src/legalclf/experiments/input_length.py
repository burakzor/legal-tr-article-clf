"""Experiment 2: how much of an article does logistic regression need to recognize the legislation type?

Usage:
    python -m legalclf.experiments.input_length --config configs/experiments/input_length_logreg.yaml

For each input length k: truncate train/dev/test to the first k words, fit TF-IDF on train,
grid-search lr x weight_decay on dev, then evaluate the best model once on test.
Outputs in `out_dir`: results.csv, per_class_f1.csv, search.csv, history.csv,
confusion_<k>.csv, input_length.png and config.json.
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

from legalclf.data import encode_labels, load_splits
from legalclf.evaluation import bootstrap_macro_f1, per_class_f1, scores
from legalclf.features import make_tfidf, truncate_words
from legalclf.models import logistic_regression as logreg
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "torch"]


def k_name(k: int | None) -> str:
    return "full" if k is None else str(k)


def run_length(k: int | None, splits: dict, y: dict, cfg: dict, device: str) -> dict:
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
    pred = logreg.predict(model, X["test"], device)
    test = scores(y["test"], pred, len(LABELS))
    low, high = bootstrap_macro_f1(splits["test"]["doc_id"], y["test"], pred, len(LABELS),
                                   n=cfg["bootstrap"]["n"], seed=cfg["seed"])
    result = {
        "k": k_name(k),
        "share_train_cut": float((splits["train"]["n_words"] > k).mean()) if k is not None else 0.0,
        "n_features": X["train"].shape[1],
        "lr": row["lr"], "weight_decay": row["weight_decay"], "best_epoch": row["best_epoch"],
        "dev_macro_f1": row["dev_macro_f1"],
        "test_macro_f1": test["macro_f1"], "test_macro_f1_low": low, "test_macro_f1_high": high,
        "test_accuracy": test["accuracy"],
    }
    f1s = per_class_f1(y["test"], pred, len(LABELS))
    confusion = pd.crosstab(pd.Categorical(np.array(LABELS)[y["test"]], categories=LABELS),
                            pd.Categorical(np.array(LABELS)[pred], categories=LABELS),
                            rownames=["true"], colnames=["pred"], dropna=False)
    return {"result": result, "per_class": dict(zip(LABELS, f1s)), "confusion": confusion,
            "search": search, "history": histories}


def plot(results: pd.DataFrame, path: Path) -> None:
    x = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(6, 4))
    yerr = [results["test_macro_f1"] - results["test_macro_f1_low"],
            results["test_macro_f1_high"] - results["test_macro_f1"]]
    ax.errorbar(x, results["test_macro_f1"], yerr=yerr, marker="o", capsize=3, label="test (95% doc bootstrap)")
    ax.plot(x, results["dev_macro_f1"], marker="s", linestyle="--", label="dev (best of search)")
    ax.set_xticks(x, results["k"])
    ax.set_xlabel("Input length (first k words of the article)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Logistic regression: effect of input length")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_splits(root, cfg["data_version"])
    y = {s: encode_labels(df["label"]).to_numpy() for s, df in splits.items()}
    print(f"device={device}  articles: " + ", ".join(f"{s}={len(df):,}" for s, df in splits.items()))

    results, per_class, search, history = [], [], [], []
    for k in cfg["input_lengths"]:
        out = run_length(k, splits, y, cfg, device)
        r = out["result"]
        print(f"k={r['k']:>4}  test macro-F1={r['test_macro_f1']:.4f} "
              f"[{r['test_macro_f1_low']:.4f}, {r['test_macro_f1_high']:.4f}]  accuracy={r['test_accuracy']:.4f}")
        results.append(r)
        per_class.append({"k": r["k"], **out["per_class"]})
        search += out["search"]
        history += out["history"]
        out["confusion"].to_csv(out_dir / f"confusion_{r['k']}.csv")

    results = pd.DataFrame(results)
    results.to_csv(out_dir / "results.csv", index=False)
    pd.DataFrame(per_class).to_csv(out_dir / "per_class_f1.csv", index=False)
    pd.DataFrame(search).to_csv(out_dir / "search.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    plot(results, out_dir / "input_length.png")
    record = {"config": cfg, "device": device, "created": date.today().isoformat(),
              "packages": {p: version(p) for p in PACKAGES}}
    (out_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
