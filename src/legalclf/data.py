"""Load the processed article dataset (data/processed/<version>/{train,dev,test}.parquet)."""
from pathlib import Path

import pandas as pd

from legalclf.preprocessing.build import LABELS, SPLITS


def load_splits(root: Path, version: str = "v1") -> dict[str, pd.DataFrame]:
    data_dir = root / "data" / "processed" / version
    splits = {s: pd.read_parquet(data_dir / f"{s}.parquet") for s in SPLITS}
    for s, df in splits.items():
        unknown = set(df["label"]) - set(LABELS)
        if unknown:
            raise ValueError(f"unknown labels in {s}: {unknown}")
    return splits


def encode_labels(labels: pd.Series) -> "pd.Series":
    """Label name -> integer id, in the fixed LABELS order (same ids for every model)."""
    return labels.map({label: i for i, label in enumerate(LABELS)}).astype("int64")
