"""Linear SVM on TF-IDF features of full articles.

Usage:
    python -m legalclf.experiments.svm --config configs/experiments/svm.yaml

Fit TF-IDF on train, grid-search C x class_weight on dev, then evaluate the best model once on test.
Outputs in `out_dir`: results.csv (dev and test metrics of the selected model), search.csv,
c_search.png, config.json and the full test report (metrics.json, per_class.csv with TP/FP/FN/TN,
confusion matrix, ROC / PR curves, score histograms, threshold trade-off).

For the feature fusion experiment, `out_dir/fusion/` holds the fitted vectorizer and model and
scores.parquet: one row per article with its decision value for every class. Dev and test scores
come from the selected model. Train scores are out-of-fold: the train set is split into folds by
document (stratified by label), and each fold is scored by a vectorizer and SVM fitted on the
other folds, so a classifier trained on these scores does not see scores of already-fitted articles.
"""
import argparse
import json
import time
from datetime import date
from importlib.metadata import version
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from legalclf.data import encode_labels, load_splits
from legalclf.evaluation import scores, write_report
from legalclf.features import make_tfidf
from legalclf.models import svm
from legalclf.plots import short
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy"]


def search(X: dict, y: dict, cfg: dict) -> tuple[list[dict], dict, object]:
    """Fit one SVM per (C, class_weight) on train; return the search log, the best row and its model."""
    rows, best = [], None
    for cw in cfg["search"]["class_weight"]:
        for C in cfg["search"]["C"]:
            start = time.time()
            model = svm.fit(X["train"], y["train"], C=float(C), class_weight=cw, seed=cfg["seed"], **cfg["model"])
            dev = scores(y["dev"], svm.predict(model, X["dev"], len(LABELS)), len(LABELS))
            row = {"C": C, "class_weight": cw, "dev_macro_f1": dev["macro_f1"], "dev_accuracy": dev["accuracy"],
                   "n_iter": int(np.max(model.n_iter_)), "seconds": round(time.time() - start, 1)}
            rows.append(row)
            print(f"  C={C:<6} class_weight={cw:<8} dev macro-F1={row['dev_macro_f1']:.4f}  "
                  f"accuracy={row['dev_accuracy']:.4f}  ({row['seconds']}s)")
            if best is None or row["dev_macro_f1"] > best[0]["dev_macro_f1"]:
                best = (row, model)
    return rows, *best


def out_of_fold_scores(texts: pd.Series, y: np.ndarray, groups: pd.Series, params: dict, cfg: dict) -> np.ndarray:
    folds = StratifiedGroupKFold(n_splits=cfg["fusion"]["folds"], shuffle=True, random_state=cfg["seed"])
    out = np.zeros((len(texts), len(LABELS)))
    for i, (fit_idx, score_idx) in enumerate(folds.split(texts, y, groups), start=1):
        vec = make_tfidf(cfg["tfidf"])
        X_fit = vec.fit_transform(texts.iloc[fit_idx])
        model = svm.fit(X_fit, y[fit_idx], seed=cfg["seed"], **params, **cfg["model"])
        out[score_idx] = svm.decision_scores(model, vec.transform(texts.iloc[score_idx]), len(LABELS))
        print(f"  fold {i}: scored {len(score_idx):,} train articles")
    return out


def save_fusion(out_dir: Path, splits: dict, split_scores: dict, vec, model) -> None:
    fusion_dir = out_dir / "fusion"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for s, df in splits.items():
        frame = df[["article_id", "doc_id", "split", "label"]].reset_index(drop=True)
        frame[[f"svm_{label}" for label in LABELS]] = split_scores[s]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(fusion_dir / "scores.parquet", index=False)
    joblib.dump(vec, fusion_dir / "vectorizer.joblib")
    joblib.dump(model, fusion_dir / "model.joblib")


def plot_search(rows: pd.DataFrame, best: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for cw, group in rows.groupby("class_weight", sort=False):
        ax.plot(group["C"], group["dev_macro_f1"], marker="o", label=f"class_weight={cw}")
    ax.scatter([best["C"]], [best["dev_macro_f1"]], s=120, facecolors="none", edgecolors="black", zorder=3,
               label="selected")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularization strength)")
    ax.set_ylabel("dev macro-F1")
    ax.set_title("Linear SVM: dev macro-F1 over C")
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

    vec = make_tfidf(cfg["tfidf"])
    X = {"train": vec.fit_transform(splits["train"]["text"])}
    X["dev"], X["test"] = vec.transform(splits["dev"]["text"]), vec.transform(splits["test"]["text"])
    print(f"TF-IDF features: {X['train'].shape[1]:,}")

    rows, best, model = search(X, y, cfg)
    rows = pd.DataFrame(rows)
    rows.to_csv(out_dir / "search.csv", index=False)
    plot_search(rows, best, out_dir / "c_search.png")
    print(f"selected: C={best['C']}  class_weight={best['class_weight']}")

    split_scores = {s: svm.decision_scores(model, X[s], len(LABELS)) for s in ("dev", "test")}
    test = write_report(out_dir, y["test"], split_scores["test"], LABELS, splits["test"]["doc_id"],
                        probabilistic=False, bootstrap_n=cfg["bootstrap"]["n"], seed=cfg["seed"],
                        title="Linear SVM, full articles, test set")
    per_class = pd.read_csv(out_dir / "per_class.csv")
    result = {"model": "linear_svm", "n_features": X["train"].shape[1], "C": best["C"],
              "class_weight": best["class_weight"], "dev_macro_f1": best["dev_macro_f1"],
              **{f"test_{m}": v for m, v in test.items() if m != "n"},
              **{f"test_f1_{name}": f for name, f in zip(short(LABELS), per_class["f1"])}}
    pd.DataFrame([result]).to_csv(out_dir / "results.csv", index=False)
    print(f"test macro-F1={test['macro_f1']:.4f} [{test['macro_f1_low']:.4f}, {test['macro_f1_high']:.4f}]  "
          f"accuracy={test['accuracy']:.4f}  balanced acc={test['balanced_accuracy']:.4f}  "
          f"macro ROC-AUC={test['macro_roc_auc']:.4f}")

    print("out-of-fold train scores for fusion")
    params = {"C": float(best["C"]), "class_weight": best["class_weight"]}
    split_scores["train"] = out_of_fold_scores(splits["train"]["text"], y["train"], splits["train"]["doc_id"],
                                               params, cfg)
    oof = scores(y["train"], split_scores["train"].argmax(axis=1), len(LABELS))
    print(f"  out-of-fold train macro-F1={oof['macro_f1']:.4f}")
    save_fusion(out_dir, splits, split_scores, vec, model)

    record = {"config": cfg, "created": date.today().isoformat(),
              "selected": {**params, "dev_macro_f1": best["dev_macro_f1"]},
              "out_of_fold_train_macro_f1": oof["macro_f1"],
              "packages": {p: version(p) for p in PACKAGES}}
    (out_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
