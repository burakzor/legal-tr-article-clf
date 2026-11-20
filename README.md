# legal-tr-article-clf

CS464 (Bilkent) Group 22 term project: predicting the type of a Turkish legislation (Law, Decree-Law, Presidential Decree, Bylaw, Presidential Regulation, Regulation, Communiqué) from a single article, using the TurkCanon task of TurkishLegalBench.

## Setup
Requires Python 3.10 or newer.

```bash
git clone git@github.com:burakzor/legal-tr-article-clf.git
cd legal-tr-article-clf
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

- `pip install -e` installs the project's own code (`src/legalclf/`) in editable mode, so `import legalclf` works from notebooks and scripts and code changes need no reinstall.
- `[dev]` adds Jupyter and pytest.
- Dependency versions are pinned in `pyproject.toml`.

## Data
- **The processed dataset is in the repo:** `data/processed/v1/` (train / dev / test parquet files). You can start working with it right away:
  ```python
  import pandas as pd
  train = pd.read_parquet("data/processed/v1/train.parquet")
  ```
- **The raw data is not in the repo.** It is only needed to rebuild the dataset.
- **Read [`data/README.md`](data/README.md) before using the data.** It covers columns, class counts, processing rules, the rules for using dev and test, and known limitations.


## Repository layout
```
configs/        settings files (YAML); preprocessing/v1.yaml builds data/processed/v1
data/           dataset card, raw data (not in git), processed versions
notebooks/      exploration and analysis notebooks; 01_data_exploration is the raw-data review
src/legalclf/   project code; preprocessing/ turns raw documents into the article dataset
tests/          pytest tests; fixtures/ holds small synthetic documents
```
