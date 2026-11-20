# Dataset card: TurkCanon articles

Article-level dataset for predicting the **type of a Turkish legislation** from a single article (madde) whose document header has been removed. It is derived from the TurkCanon task of TurkishLegalBench.

## Source and license
- **Source:** TurkishLegalBench, TurkCanon task, latest version.
  - Zenodo: [10.5281/zenodo.20680163](https://doi.org/10.5281/zenodo.20680163)
  - Authors: Erkan & Yozgatligil (METU)
- **License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). The processed data in this folder is a derivative work and is shared under the same license. When you use or share it:
  - credit the original dataset;
  - do not use it commercially;
  - share derivatives under the same license.

## Task and labels
Each example is one article; the label is the type of the legislation the article comes from. The 7 types, by their level in the hierarchy of norms:

| Label | Type | Level |
|---|---|---|
| `KANUN` | Law | 1 |
| `KANUN_HUKMUNDE_KARARNAME` | Decree-Law (KHK) | 1 |
| `CUMHURBASKANLIGI_KARARNAMESI` | Presidential Decree | 1 |
| `TUZUK` | Bylaw | 2 |
| `CUMHURBASKANLIGI_YONETMELIGI` | Presidential Regulation | 2 |
| `YONETMELIK` | Regulation | 3 |
| `TEBLIG` | Communiqué | 4 |

## Folder structure
```
data/
├── raw/TurkCanon/        original files (not in git; see "Raw data")
└── processed/v1/         article-level dataset, version 1 (in git)
```

## Raw data
- **Files:** `train.jsonl`, `dev.jsonl`, `test.jsonl` in `data/raw/TurkCanon/`.
- **Not in git:** `train.jsonl` is 185 MB, over GitHub's file size limit. Download the TurkCanon files from the Zenodo record above and place them in `data/raw/TurkCanon/`.
- **Format:** one JSON object per line, one document per line. Fields: `doc_id`, `source` (always `legislative`), `text`, `label`.

| Split | Documents |
|---|---|
| train | 6,385 |
| dev | 908 |
| test | 1,833 |

- **Splits:** the official split is document-level and stratified; class proportions are nearly equal across splits.
- **Class imbalance:** `YONETMELIK` is 44% and `TEBLIG` 40% of documents. `TUZUK`, `KANUN_HUKMUNDE_KARARNAME` and `CUMHURBASKANLIGI_KARARNAMESI` are each under 2%. Dev has only 3–10 documents of these classes.
- **`[MASK]`:** the dataset authors replaced words that name the legislation type with `[MASK]`, including their suffixes ("Bu Yönetmeliğin amacı" → "Bu [MASK] amacı").

## Versions
| Version | Folder | Status |
|---|---|---|
| v1 | `data/processed/v1/` | Frozen 2026-11-20. Built from the raw files whose md5 are in `config.json` |

- **v1 is never rebuilt in place.** Any later change to the preprocessing code or config is written to a new folder (`v2`, …), so results computed on v1 stay reproducible.
- **v1 was shared before the planned manual check** of 100 random articles. If that check leads to rule changes, they will go to v2.

## Processed data (v1)
```python
import pandas as pd
train = pd.read_parquet("data/processed/v1/train.parquet")
```

| File | Size | Content |
|---|---|---|
| `train.parquet` | 20.9 MB | training articles |
| `dev.parquet` | 3.1 MB | validation articles |
| `test.parquet` | 6.4 MB | test articles |
| `stats.csv` | 4 KB | documents and articles per split × label after each preprocessing step |
| `config.json` | 1 KB | settings, seed, md5 of the raw files and package versions used for this build |
| `manual_check.csv` | 0.2 MB | 100 random articles (raw and cleaned text) for the manual check of the cleaning rules |

**Columns** (one row = one article):

| Column | Type | Description |
|---|---|---|
| `article_id` | str | `<doc_id>_<idx>`, where `idx` is the position of the article in its document (e.g. `LEG-000516ba9767_0003`) |
| `doc_id` | str | source document; use it for document-level analysis and bootstrap |
| `split` | str | `train` / `dev` / `test` |
| `label` | str | legislation type (see "Task and labels") |
| `article_kind` | str | `normal`, `temporary` (GEÇİCİ / MUVAKKAT), `additional` (EK), `repeated` (MÜKERRER) |
| `article_number` | int | the number in the article marker |
| `text` | str | cleaned article text, the model input |
| `n_words` | int | words in `text` |
| `truncated` | bool | `text` was cut to the first 1,000 words |

**Articles per class:**

| Label | train | dev | test |
|---|---|---|---|
| `YONETMELIK` | 40,764 | 5,905 | 11,822 |
| `TEBLIG` | 11,903 | 1,689 | 3,336 |
| `KANUN` | 8,110 | 1,179 | 2,357 |
| `CUMHURBASKANLIGI_YONETMELIGI` | 3,698 | 502 | 1,085 |
| `TUZUK` | 1,233 | 166 | 393 |
| `KANUN_HUKMUNDE_KARARNAME` | 496 | 72 | 129 |
| `CUMHURBASKANLIGI_KARARNAMESI` | 292 | 24 | 114 |
| **Total** | **66,496** | **9,537** | **19,236** |
| Documents | 5,146 | 744 | 1,496 |

Article length: median ~70 words, 90th percentile ~280 words, maximum 1,000 (truncated). 0.7% of articles are truncated.

**Effect of each step** (articles; the `raw` row counts documents; details per class in `stats.csv`):

| Step | train | dev | test |
|---|---|---|---|
| raw (documents) | 6,385 | 908 | 1,833 |
| split into articles | 137,382 | 18,352 | 38,551 |
| clean + filter | 124,038 | 16,503 | 34,846 |
| deduplicate | 118,322 | 15,444 | 32,217 |
| per-document cap | 66,496 | 9,537 | 19,236 |

## Processing rules (v1)
Evidence for each rule is in `notebooks/01_data_exploration.ipynb`; the code is in `src/legalclf/preprocessing/`.

1. **Unit = article.** Documents are split at article markers (`MADDE 1 –`, `Madde 5/A-`, `GEÇİCİ MADDE 2 –`, `Muvakkat Madde 1 -`, …, six dash characters). The document header before the first marker is dropped.
2. **Documents without any article marker are excluded:** 1,630 documents, 99% of them `TEBLIG` / `YONETMELIK` annexes, forms and reports.
3. **Cleaning** removes what is not the authored text of the article:
   - the `MADDE n –` prefix;
   - section headings and the next article's heading stuck to the end of an article ("… belirlemektir. Kapsam");
   - amendment notes (`(Değişik:RG-…)`, `(Ek: …)`, `(Mülga …: …)`);
   - web artifacts ("Ekleri için tıklayınız" and the Resmî Gazete table after it);
   - `[MASK]` and paragraph numbers `(1)`, `(2)`. References such as "(1) sayılı cetvel" are kept, and so are bullets `a)`, `b)`.
4. Articles longer than **1,000 words** are truncated.
5. **Filters:** boilerplate articles ("… yayımı tarihinde yürürlüğe girer", "… Bakanı yürütür", fewer than 60 words) and articles shorter than 5 words are dropped.
6. **Deduplication** on a normalized key (lowercase, collapsed spaces, digits → 0):
   - texts that occur with more than one label are dropped everywhere;
   - one copy is kept within each split;
   - dev/test articles whose text occurs in train, and test articles whose text occurs in dev, are dropped. **The test set contains only unseen texts.**
7. **Per-document cap:** at most 20 random articles per document in every split (seed 42), so a few long documents do not dominate a class.
8. **Class balance is not changed.** All splits keep the natural distribution; balancing belongs to training.

## Usage rules for the team
- Tune hyperparameters **only on `dev`**. Use `test` once, for the final evaluation of the selected configurations.
- Evaluate every model on the same `v1/test.parquet`.
- The main metric is macro-F1. For confidence intervals, resample **documents** (`doc_id`), not articles: articles of one document are not independent.
- Fit anything learned from text (vocabulary, IDF, SVD, scalers) on `train` only.

## How to build
```bash
python -m legalclf.preprocessing.build --config configs/preprocessing/v1.yaml
```
- Every setting is in the config file. Paths are relative to the repo root.
- The build needs the raw files (see "Raw data") and the package installed (`pip install -e ".[dev]"`).
- It is deterministic: the same config and raw files give byte-identical parquet files.
- If the output folder already holds data built with different settings or raw files, the build stops instead of overwriting it.
- It also writes `manual_check.csv`: 100 random articles, raw and cleaned text side by side, with empty `ok` / `note` columns for the manual check.

## Variants
None yet. Planned: `v1_keep-paragraphs`, the same as v1 but with paragraph numbers kept, to measure how much this formatting cue helps.

## Known limitations
- **Excluded documents.** The 1,630 documents without article markers (mostly `TEBLIG` annexes and forms) are excluded, so the dataset does not cover those texts.
- **Very small classes in dev.** `CUMHURBASKANLIGI_KARARNAMESI` has 3 documents / 24 articles and `KANUN_HUKMUNDE_KARARNAME` has 6 documents / 72 articles, so dev scores for these classes are noisy.
- **Missing article numbers.** Many `KANUN` and `KHK` documents have gaps in their article numbering. These are mostly repealed articles that are absent from the source text, not splitting errors.
- **Known leftovers:**
  - 6% of articles do not end with punctuation: headings longer than 12 words, or list items without final punctuation;
  - 0.6% still contain a section heading;
  - a few last articles still carry an annex;
  - some tax communiqués quote articles of laws.
- **Masking in the source data is aggressive.** Some words that do not name a legislation type are masked too, and the masks are removed with them.
- **Manual check pending.** The cleaning rules have not yet been checked by hand on a random sample.
