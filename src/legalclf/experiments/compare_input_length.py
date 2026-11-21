"""Experiment 2: compare logistic regression, Naive Bayes and linear SVM across input lengths.

Usage:
    python -m legalclf.experiments.compare_input_length --config configs/experiments/compare_input_length.yaml

Reads the saved outputs of the three input length runs (results.csv, per_class_f1.csv and
k_<k>/confusion_counts.csv of each model) and writes tables and figures only. Nothing is
trained and no new evaluation is done, so every number here is the one in the model folders.
All three runs use the same data (v1), input lengths, TF-IDF settings and test set.

Outputs in `out_dir`:
  combined_results.csv          every test/dev metric of every model at every k (long format)
  table_macro_f1.csv            test macro-F1 with its 95% document-bootstrap interval, k x model
  table_relative_to_full.csv    test macro-F1 at k divided by the model's full-article macro-F1
  table_words_needed.csv        smallest k that reaches each threshold share of the full-article score
  table_per_class_f1.csv        test F1 per class, model x k
  table_hyperparameters.csv     hyperparameters selected on dev at every k
  table_dev_vs_test.csv         dev and test macro-F1 of the selected configuration
  fig_macro_f1_vs_length.png    test macro-F1 vs k, three models, with 95% intervals
  fig_per_class_f1_vs_length.png  one panel per class: test F1 vs k, three models
  fig_per_class_f1_full.png     test F1 per class on full articles, grouped bars
  fig_confusion_matrices.png    row-normalized test confusion matrices, model x selected k
  fig_dev_vs_test.png           dev and test macro-F1 vs k per model
"""
import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

from legalclf.plots import short
from legalclf.preprocessing.build import LABELS, find_repo_root

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the shared chart palette (validated for colour-vision deficiency).
# Each model also has its own marker and line style, so identity never rests on colour alone.
STYLE = [
    {"color": "#2a78d6", "marker": "o", "linestyle": "-"},
    {"color": "#eb6834", "marker": "s", "linestyle": "--"},
    {"color": "#1baf7a", "marker": "^", "linestyle": "-."},
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e6e5e0"

plt.rcParams.update({
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False, "font.size": 10,
    "lines.linewidth": 2, "lines.markersize": 6,
})


def load_model(name: str, folder: Path, k_order: list[str] | None) -> dict:
    results = pd.read_csv(folder / "results.csv", dtype={"k": str})
    per_class = pd.read_csv(folder / "per_class_f1.csv", dtype={"k": str})
    if k_order is not None and list(results["k"]) != k_order:
        raise ValueError(f"{name}: input lengths {list(results['k'])} differ from {k_order}")
    return {"name": name, "folder": folder, "results": results, "per_class": per_class}


def combined_long(models: list[dict]) -> pd.DataFrame:
    frames = []
    for m in models:
        r = m["results"]
        metric_cols = [c for c in r.columns if c.startswith(("test_", "dev_"))]
        long = r.melt(id_vars="k", value_vars=metric_cols, var_name="metric", value_name="value")
        long.insert(0, "model", m["name"])
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def table_macro_f1(models: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame({"k": models[0]["results"]["k"]})
    for m in models:
        r = m["results"]
        out[f"{m['name']}"] = r["test_macro_f1"].round(4)
        out[f"{m['name']} low"] = r["test_macro_f1_low"].round(4)
        out[f"{m['name']} high"] = r["test_macro_f1_high"].round(4)
    return out


def table_relative(models: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame({"k": models[0]["results"]["k"]})
    for m in models:
        r = m["results"]
        full = r.loc[r["k"] == "full", "test_macro_f1"].item()
        out[m["name"]] = (r["test_macro_f1"] / full).round(4)
    return out


def table_words_needed(models: list[dict], thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for m in models:
        r = m["results"]
        full = r.loc[r["k"] == "full", "test_macro_f1"].item()
        row = {"model": m["name"], "full_article_test_macro_f1": round(full, 4)}
        for t in thresholds:
            reached = r[r["test_macro_f1"] >= t * full]
            row[f"words_for_{int(t * 100)}pct"] = reached["k"].iloc[0]   # rows are in increasing k
        rows.append(row)
    return pd.DataFrame(rows)


def table_per_class(models: list[dict]) -> pd.DataFrame:
    frames = []
    for m in models:
        pc = m["per_class"][["k", *LABELS]].copy()
        pc.insert(0, "model", m["name"])
        frames.append(pc)
    out = pd.concat(frames, ignore_index=True)
    out[LABELS] = out[LABELS].round(4)
    return out


def table_hyperparameters(models: list[dict], hp: dict) -> pd.DataFrame:
    frames = []
    for m in models:
        cols = hp[m["name"]]
        t = m["results"][["k", *cols]].copy()
        t.insert(0, "model", m["name"])
        t[cols] = t[cols].astype(object).where(t[cols].notna(), "-")   # e.g. fit_prior does not apply to ComplementNB
        t["selected"] = t[cols].astype(str).agg(lambda x: ", ".join(f"{c}={v}" for c, v in zip(cols, x)), axis=1)
        frames.append(t[["model", "k", "selected"]])
    return pd.concat(frames, ignore_index=True)


def table_dev_vs_test(models: list[dict]) -> pd.DataFrame:
    frames = []
    for m in models:
        r = m["results"]
        frames.append(pd.DataFrame({"model": m["name"], "k": r["k"], "dev_macro_f1": r["dev_macro_f1"].round(4),
                                    "test_macro_f1": r["test_macro_f1"].round(4),
                                    "test_minus_dev": (r["test_macro_f1"] - r["dev_macro_f1"]).round(4)}))
    return pd.concat(frames, ignore_index=True)


def plot_macro_f1(models: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    for m, st in zip(models, STYLE):
        r = m["results"]
        x = np.arange(len(r))
        ax.fill_between(x, r["test_macro_f1_low"], r["test_macro_f1_high"], color=st["color"], alpha=0.12, linewidth=0)
        ax.plot(x, r["test_macro_f1"], label=m["name"], **st)
        ax.annotate(m["name"], (x[-1], r["test_macro_f1"].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xticks(x, models[0]["results"]["k"])
    ax.set_xlabel("Input length (first k words of the article)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Test macro-F1 by input length (bands: 95% document-bootstrap interval)", fontsize=10)
    ax.set_xlim(-0.3, len(x) - 1 + 1.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_class_vs_length(models: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    k = models[0]["per_class"]["k"]
    x = np.arange(len(k))
    for ax, label, name in zip(axes.flat, LABELS, short(LABELS)):
        for m, st in zip(models, STYLE):
            ax.plot(x, m["per_class"][label], label=m["name"], markersize=4, **st)
        ax.set_title(f"{name}  ({label.replace('_', ' ').title()})", fontsize=8.5)
        ax.set_xticks(x, k, fontsize=8)
        ax.set_ylim(0, 1)
    last = axes.flat[-1]
    last.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    last.legend(handles, labels, loc="center", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("Input length (k words)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Test F1")
    fig.suptitle("Per-class test F1 by input length", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_per_class_full(models: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(LABELS))
    width = 0.26
    for i, (m, st) in enumerate(zip(models, STYLE)):
        f1 = m["per_class"].loc[m["per_class"]["k"] == "full", LABELS].iloc[0].to_numpy()
        ax.bar(x + (i - 1) * width, f1, width - 0.03, color=st["color"], label=m["name"])
    ax.set_xticks(x, short(LABELS))
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class test F1 on full articles", fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_confusions(models: list[dict], lengths: list[str], path: Path) -> None:
    fig, axes = plt.subplots(len(lengths), len(models), figsize=(4.2 * len(models), 3.9 * len(lengths)))
    axes = np.atleast_2d(axes)
    names = short(LABELS)
    for i, k in enumerate(lengths):
        for j, m in enumerate(models):
            counts = pd.read_csv(m["folder"] / f"k_{k}" / "confusion_counts.csv", index_col=0)
            counts = counts.loc[LABELS, LABELS].to_numpy(dtype=float)
            norm = counts / counts.sum(axis=1, keepdims=True)
            ax = axes[i, j]
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            for a in range(len(LABELS)):
                for b in range(len(LABELS)):
                    ax.text(b, a, f"{norm[a, b]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if norm[a, b] > 0.55 else INK)
            ax.set_xticks(range(len(LABELS)), names, fontsize=7, rotation=45)
            ax.set_yticks(range(len(LABELS)), names, fontsize=7)
            ax.grid(False)
            ax.set_title(f"{m['name']}, {'full articles' if k == 'full' else f'first {k} words'}", fontsize=9)
            if j == 0:
                ax.set_ylabel("True type")
            if i == len(lengths) - 1:
                ax.set_xlabel("Predicted type")
    fig.suptitle("Test confusion matrices (rows normalized: diagonal = recall)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_dev_vs_test(models: list[dict], path: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
    for ax, m, st in zip(axes, models, STYLE):
        r = m["results"]
        x = np.arange(len(r))
        ax.plot(x, r["test_macro_f1"], label="test", **st)
        ax.plot(x, r["dev_macro_f1"], label="dev (selection)", color=MUTED, marker=st["marker"], linestyle=":")
        ax.set_xticks(x, r["k"], fontsize=8)
        ax.set_title(m["name"], fontsize=10)
        ax.set_xlabel("Input length (k words)")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Macro-F1")
    fig.suptitle("Dev vs test macro-F1 of the configuration selected on dev", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = find_repo_root(Path.cwd())
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out_dir = root / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    models, k_order = [], None
    for name, folder in cfg["models"].items():
        m = load_model(name, root / folder, k_order)
        k_order = list(m["results"]["k"])
        models.append(m)
    print("models: " + ", ".join(m["name"] for m in models) + f"  |  input lengths: {k_order}")

    combined_long(models).to_csv(out_dir / "combined_results.csv", index=False)
    tables = {
        "table_macro_f1.csv": table_macro_f1(models),
        "table_relative_to_full.csv": table_relative(models),
        "table_words_needed.csv": table_words_needed(models, cfg["thresholds"]),
        "table_per_class_f1.csv": table_per_class(models),
        "table_hyperparameters.csv": table_hyperparameters(models, cfg["hyperparameters"]),
        "table_dev_vs_test.csv": table_dev_vs_test(models),
    }
    for file, table in tables.items():
        table.to_csv(out_dir / file, index=False)

    plot_macro_f1(models, out_dir / "fig_macro_f1_vs_length.png")
    plot_per_class_vs_length(models, out_dir / "fig_per_class_f1_vs_length.png")
    plot_per_class_full(models, out_dir / "fig_per_class_f1_full.png")
    plot_confusions(models, [str(k) for k in cfg["confusion_lengths"]], out_dir / "fig_confusion_matrices.png")
    plot_dev_vs_test(models, out_dir / "fig_dev_vs_test.png")

    record = {"config": cfg, "created": date.today().isoformat()}
    (out_dir / "config.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(tables["table_macro_f1.csv"][["k", *cfg["models"]]].to_string(index=False))
    print(tables["table_words_needed.csv"].to_string(index=False))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
