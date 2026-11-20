"""Build the article-level dataset from the raw TurkCanon documents.

Usage:
    python -m legalclf.preprocessing.build --config configs/preprocessing/v1.yaml

Steps (see CLAUDE.md, "Preprocessing decisions"): read -> split -> clean and filter ->
deduplicate -> per-document cap -> save. Outputs go to `paths.out_dir`:
{train,dev,test}.parquet, stats.csv, config.json and manual_check.csv.
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from legalclf.preprocessing.clean import CleaningConfig, clean_article, normalize_whitespace
from legalclf.preprocessing.filters import is_boilerplate, is_too_short, normalized_key
from legalclf.preprocessing.segment import split_document

SPLITS = ["train", "dev", "test"]
LABELS = ["YONETMELIK", "TEBLIG", "KANUN", "CUMHURBASKANLIGI_YONETMELIGI", "TUZUK",
          "KANUN_HUKMUNDE_KARARNAME", "CUMHURBASKANLIGI_KARARNAMESI"]
OUTPUT_COLUMNS = ["article_id", "doc_id", "split", "label", "article_kind", "article_number",
                  "text", "n_words", "truncated"]
PACKAGES = ["numpy", "pandas", "pyarrow", "pyyaml"]


class ConfigMismatchError(RuntimeError):
    """The output folder already holds data built with different settings."""


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError(f"no pyproject.toml above {start}")


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    for key in ("seed", "paths", "cleaning", "filters", "cap", "manual_check"):
        if key not in cfg:
            raise ValueError(f"config is missing '{key}'")
    CleaningConfig(**cfg["cleaning"])
    if set(cfg["cap"]) != set(SPLITS) or any(int(v) < 1 for v in cfg["cap"].values()):
        raise ValueError(f"cap must give a positive value for each of {SPLITS}")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for split in SPLITS:
        with open(raw_dir / f"{split}.jsonl", encoding="utf-8") as f:
            rows += [{**json.loads(line), "split": split} for line in f if line.strip()]
    docs = pd.DataFrame(rows)[["split", "doc_id", "label", "text"]]
    unknown = set(docs["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels in raw data: {unknown}")
    return docs


def split_articles(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d in docs.itertuples(index=False):
        _, articles = split_document(d.text)
        rows += [{"split": d.split, "doc_id": d.doc_id, "label": d.label, "idx": a.idx,
                  "article_kind": a.kind, "article_number": a.number, "raw": a.raw} for a in articles]
    return pd.DataFrame(rows)


def clean_and_filter(art: pd.DataFrame, cleaning: CleaningConfig, filters: dict) -> pd.DataFrame:
    cleaned = [clean_article(raw, cleaning) for raw in art["raw"]]
    art = art.assign(text=[t for t, _ in cleaned], truncated=[tr for _, tr in cleaned])
    art["n_words"] = art["text"].str.split().str.len()
    drop = [is_too_short(t, filters["min_words"]) or is_boilerplate(t, filters["boilerplate_max_words"])
            for t in art["text"]]
    return art[~np.array(drop)]


def deduplicate(art: pd.DataFrame) -> pd.DataFrame:
    """Drop multi-label texts, keep one copy per split, then remove dev/test texts seen in earlier splits."""
    art = art.assign(key=art["text"].map(normalized_key),
                     split_order=art["split"].map(SPLITS.index))
    multi = art.groupby("key")["label"].nunique()
    art = art[~art["key"].isin(multi[multi > 1].index)]
    art = art.sort_values(["split_order", "doc_id", "idx"]).drop_duplicates(["split", "key"])
    seen = set()
    keep = []
    for split in SPLITS:
        part = art[art["split"] == split]
        part = part[~part["key"].isin(seen)]
        seen |= set(part["key"])
        keep.append(part)
    return pd.concat(keep).drop(columns="split_order")


def doc_rng(seed: int, doc_id: str) -> np.random.Generator:
    """Per-document generator: a document's sample does not depend on the other documents."""
    return np.random.default_rng([seed, int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)])


def cap_per_document(art: pd.DataFrame, caps: dict, seed: int) -> pd.DataFrame:
    parts = []
    for (split, doc_id), g in art.sort_values(["split", "doc_id", "idx"]).groupby(["split", "doc_id"], sort=True):
        cap = int(caps[split])
        if len(g) > cap:
            g = g.iloc[np.sort(doc_rng(seed, doc_id).choice(len(g), size=cap, replace=False))]
        parts.append(g)
    return pd.concat(parts)


def counts(step: str, art: pd.DataFrame | None, docs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Documents and articles per split x label after a step."""
    grid = pd.MultiIndex.from_product([SPLITS, LABELS], names=["split", "label"])
    if art is None:
        out = pd.DataFrame({"docs": docs.groupby(["split", "label"]).size()}).reindex(grid, fill_value=0)
        out["articles"] = np.nan  # the raw data has no articles yet
    else:
        g = art.groupby(["split", "label"])
        out = pd.DataFrame({"docs": g["doc_id"].nunique(), "articles": g.size()}).reindex(grid, fill_value=0)
    return out.reset_index().assign(step=step)


def make_stats(snapshots: list[pd.DataFrame]) -> pd.DataFrame:
    stats = pd.concat(snapshots, ignore_index=True)
    stats["articles"] = stats["articles"].astype(float)
    stats["removed_articles"] = -stats.groupby(["split", "label"])["articles"].diff()
    cols = ["step", "split", "label", "docs", "articles", "removed_articles"]
    return stats[cols].astype({"docs": "Int64", "articles": "Int64", "removed_articles": "Int64"})


def check_out_dir(out_dir: Path, record: dict) -> None:
    previous = out_dir / "config.json"
    if not previous.exists():
        return
    old = json.loads(previous.read_text(encoding="utf-8"))
    if old.get("config") != record["config"] or old.get("raw_files") != record["raw_files"]:
        raise ConfigMismatchError(
            f"{out_dir} already holds data built with different settings or raw files; "
            "use another out_dir or delete the folder first")


def save(final: pd.DataFrame, stats: pd.DataFrame, record: dict, cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = final.assign(article_id=final["doc_id"] + "_" + final["idx"].map("{:04d}".format))
    final = final.sort_values("article_id").reset_index(drop=True)
    for split in SPLITS:
        part = final.loc[final["split"] == split, OUTPUT_COLUMNS].reset_index(drop=True)
        part.to_parquet(out_dir / f"{split}.parquet", index=False, compression="zstd")
    stats.to_csv(out_dir / "stats.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    n = min(int(cfg["manual_check"]["n"]), len(final))
    sample = final.iloc[np.sort(np.random.default_rng(cfg["seed"]).choice(len(final), size=n, replace=False))]
    check = sample.assign(raw=sample["raw"].map(normalize_whitespace), ok="", note="")
    check[["article_id", "split", "label", "raw", "text", "ok", "note"]].to_csv(
        out_dir / "manual_check.csv", index=False)


def build(cfg: dict, root: Path) -> pd.DataFrame:
    validate_config(cfg)
    raw_dir = root / cfg["paths"]["raw_dir"]
    out_dir = root / cfg["paths"]["out_dir"]
    record = {
        "config": cfg,
        "raw_files": {f"{s}.jsonl": md5_file(raw_dir / f"{s}.jsonl") for s in SPLITS},
        "packages": {p: version(p) for p in PACKAGES},
        "created": date.today().isoformat(),
    }
    check_out_dir(out_dir, record)

    docs = read_raw(raw_dir)
    snapshots = [counts("raw", None, docs)]
    art = split_articles(docs)
    snapshots.append(counts("split", art))
    art = clean_and_filter(art, CleaningConfig(**cfg["cleaning"]), cfg["filters"])
    snapshots.append(counts("clean_filter", art))
    art = deduplicate(art)
    snapshots.append(counts("dedupe", art))
    art = cap_per_document(art, cfg["cap"], cfg["seed"])
    snapshots.append(counts("cap", art))

    stats = make_stats(snapshots)
    save(art, stats, record, cfg, out_dir)
    print_summary(stats, out_dir)
    return art


def print_summary(stats: pd.DataFrame, out_dir: Path) -> None:
    total = stats.groupby(["step", "split"], sort=False)[["docs", "articles"]].sum().unstack("split")
    print(f"written to {out_dir}\n\ndocuments / articles after each step:")
    print(total.to_string())
    final = stats[stats["step"] == "cap"].pivot(index="label", columns="split", values="articles")[SPLITS]
    print("\nfinal articles per class:")
    print(final.loc[LABELS].to_string())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        build(load_config(config_path), find_repo_root(config_path.parent))
    except ConfigMismatchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
