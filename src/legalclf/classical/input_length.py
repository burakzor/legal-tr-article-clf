"""Experiment 2: effect of input length on the classical TF-IDF models.

For every input length k (first k words of the article), train/dev/test texts are cut to
k words, TF-IDF is fit on train only, hyperparameters are chosen by dev macro-F1, and the
selected configuration is evaluated once on test with a document-level bootstrap CI.

Usage:
    python -m legalclf.classical.input_length --config configs/experiments/input_length_nb.yaml
    python -m legalclf.classical.input_length --config ... --dev-only   # tune, do not touch test

Adding a model (e.g. LR or SVM): write a `<name>_candidates(grid)` function that yields
(params, estimator) pairs, register it in MODELS, and make a config with `model: <name>`.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import matplotlib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from legalclf.classical.common import (FULL, doc_bootstrap_ci, load_splits, macro_f1,
                                       per_class_f1, truncate_texts, turkish_lower)
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def grid_points(grid: dict) -> list[dict]:
    """All combinations of a {param: [values]} grid."""
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def naive_bayes_candidates(grid: dict):
    for p in grid_points(grid):
        if p["variant"] == "multinomial":
            yield p, MultinomialNB(alpha=p["alpha"], fit_prior=p.get("fit_prior", True))
        elif p["variant"] == "complement":
            # ComplementNB does not use class priors, so fit_prior would only duplicate runs
            if p.get("fit_prior", False) is False:
                yield {**p, "fit_prior": None}, ComplementNB(alpha=p["alpha"])
        else:
            raise ValueError(f"unknown naive_bayes variant {p['variant']!r}")


MODELS = {"naive_bayes": naive_bayes_candidates}


def make_vectorizer(fixed: dict, params: dict) -> TfidfVectorizer:
    kw = {**fixed, **params}
    kw["ngram_range"] = tuple(kw["ngram_range"])
    return TfidfVectorizer(preprocessor=turkish_lower, **kw)


def length_label(k) -> str:
    return FULL if k == FULL else str(int(k))


def run_length(k, data, cfg, dev_only: bool) -> tuple[list[dict], dict | None, pd.DataFrame | None]:
    texts = {s: truncate_texts(df["text"], k) for s, df in data.items()}
    y = {s: df["label"].to_numpy() for s, df in data.items()}
    candidates = MODELS[cfg["model"]]

    rows, best = [], None
    for vec_params in grid_points(cfg["tfidf"]["grid"]):
        vec = make_vectorizer(cfg["tfidf"]["fixed"], vec_params)
        x_train = vec.fit_transform(texts["train"])
        x_dev = vec.transform(texts["dev"])
        for model_params, model in candidates(cfg["model_grid"]):
            t0 = time.perf_counter()
            model.fit(x_train, y["train"])
            score = macro_f1(y["dev"], model.predict(x_dev))
            row = {"length": length_label(k), **{f"tfidf_{p}": str(v) for p, v in vec_params.items()},
                   **model_params, "n_features": x_train.shape[1], "dev_macro_f1": score,
                   "fit_seconds": round(time.perf_counter() - t0, 2)}
            rows.append(row)
            if best is None or score > best["dev_macro_f1"]:
                best = {"dev_macro_f1": score, "vec_params": vec_params,
                        "model_params": model_params, "vec": vec, "model": model}

    print(f"  k={length_label(k):>4}: best dev macro-F1 {best['dev_macro_f1']:.4f} "
          f"with {best['vec_params']} {best['model_params']}")
    if dev_only:
        return rows, None, None

    # best["vec"] / best["model"] were fit on train only; test is used once, here.
    test = data["test"]
    pred = best["model"].predict(best["vec"].transform(texts["test"]))
    lo, hi = doc_bootstrap_ci(y["test"], pred, test["doc_id"], cfg["bootstrap"]["n"],
                              cfg["seed"], cfg["bootstrap"]["level"])
    result = {
        "length": length_label(k),
        "share_test_articles_cut": float((test["n_words"] > int(k)).mean()) if k != FULL else 0.0,
        "mean_test_input_words": float(texts["test"].str.split().str.len().mean()),
        "dev_macro_f1": best["dev_macro_f1"],
        "test_macro_f1": macro_f1(y["test"], pred),
        "test_ci_low": lo, "test_ci_high": hi,
        **{f"test_f1_{l}": v for l, v in per_class_f1(y["test"], pred).items()},
        "tfidf_params": json.dumps(best["vec_params"]),
        "model_params": json.dumps(best["model_params"]),
    }
    preds = pd.DataFrame({"length": length_label(k), "article_id": test["article_id"],
                          "doc_id": test["doc_id"], "label": y["test"], "pred": pred})
    return rows, result, preds


def plot(results: pd.DataFrame, title: str, path: Path) -> None:
    x = range(len(results))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, results["dev_macro_f1"], "o--", color="0.6", label="dev (selection)")
    ax.plot(x, results["test_macro_f1"], "o-", label="test")
    ax.fill_between(x, results["test_ci_low"], results["test_ci_high"], alpha=0.2,
                    label="test 95% CI (document bootstrap)")
    ax.set_xticks(list(x), results["length"])
    ax.set_xlabel("input length (first k words of the article)")
    ax.set_ylabel("macro-F1")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dev-only", action="store_true",
                        help="only tune on dev; do not evaluate on test")
    args = parser.parse_args()

    root = find_repo_root(Path.cwd())
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if cfg["model"] not in MODELS:
        raise ValueError(f"model must be one of {sorted(MODELS)}")
    out_dir = root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    data = load_splits(root / cfg["data_dir"])
    print(f"{cfg['model']}: lengths {cfg['lengths']}")
    grid_rows, results, preds = [], [], []
    for k in cfg["lengths"]:
        rows, result, pred = run_length(k, data, cfg, args.dev_only)
        grid_rows += rows
        if result is not None:
            results.append(result)
            preds.append(pred)

    pd.DataFrame(grid_rows).to_csv(out_dir / "dev_grid.csv", index=False)
    if args.dev_only:
        print(f"dev grid written to {out_dir / 'dev_grid.csv'}")
        return
    results = pd.DataFrame(results)
    results.to_csv(out_dir / "results.csv", index=False)
    pd.concat(preds).to_parquet(out_dir / "test_predictions.parquet", index=False)
    plot(results, f"{cfg['model']}: macro-F1 vs input length", out_dir / "macro_f1_vs_length.png")
    cols = ["length", "share_test_articles_cut", "dev_macro_f1", "test_macro_f1",
            "test_ci_low", "test_ci_high"]
    print(results[cols].round(4).to_string(index=False))
    print(f"outputs in {out_dir}")


if __name__ == "__main__":
    main()
