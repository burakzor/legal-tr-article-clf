"""End-to-end test of the build pipeline on the synthetic documents in fixtures/mini.jsonl."""
import json
from pathlib import Path

import pandas as pd
import pytest

from legalclf.preprocessing.build import SPLITS, ConfigMismatchError, build
from legalclf.preprocessing.filters import normalized_key

FIXTURE = Path(__file__).parent / "fixtures" / "mini.jsonl"
CAPS = {"train": 5, "dev": 20, "test": 20}


def make_raw(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    lines = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    for split in SPLITS:
        rows = [{k: v for k, v in r.items() if k != "split"} for r in lines if r["split"] == split]
        (raw / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return raw


def make_cfg(raw, out, **cap):
    return {
        "seed": 42,
        "paths": {"raw_dir": str(raw), "out_dir": str(out)},
        "cleaning": {"mask": "remove", "mask_placeholder": None, "paragraph_numbers": "remove", "max_words": 1000},
        "filters": {"min_words": 5, "boilerplate_max_words": 60},
        "cap": {**CAPS, **cap},
        "manual_check": {"n": 3},
    }


@pytest.fixture
def built(tmp_path):
    raw = make_raw(tmp_path)
    out = tmp_path / "out"
    build(make_cfg(raw, out), tmp_path)
    data = {s: pd.read_parquet(out / f"{s}.parquet") for s in SPLITS}
    return raw, out, data


def texts(df, doc_id):
    return df.loc[df["doc_id"] == doc_id, "text"].tolist()


def test_expected_articles(built):
    _, _, d = built
    # TR-Y1: articles 4-5 are boilerplate; article 3 is kept here and dropped from TR-Y2 as a duplicate
    assert texts(d["train"], "TR-Y1") == [
        "Bu amacı; örnek kurulun çalışma usul ve esaslarını belirlemektir.",
        "Bu; kamu kurumlarında görev yapan kurulları kapsar.",
        "Kurum başvuruları elektronik ortamda alır ve on gün içinde sonuçlandırır.",
    ]
    # "Komisyon üyeleri ..." occurs as YONETMELIK and TEBLIG -> dropped everywhere
    assert texts(d["train"], "TR-Y2") == []
    assert texts(d["train"], "TR-T2") == ["İthalatçı firmalar beyannameyi elektronik ortamda gümrük idaresine sunar."]
    assert "TR-T1" not in set(d["train"]["doc_id"])  # no article marker
    assert texts(d["dev"], "DV-Y1") == ["Denetim raporları her yıl mart ayı sonuna kadar Bakanlığa gönderilir."]
    # article 1 occurs in dev, article 2 in train
    assert texts(d["test"], "TS-Y1") == ["Kurul yılda en az dört kez toplanır ve gündemi önceden duyurur."]
    z1 = d["test"][d["test"]["doc_id"] == "TS-Z1"]
    assert z1["article_kind"].tolist() == ["normal", "temporary", "normal"]
    assert z1["text"].iloc[2] == "Bu iş 5 inci madde uyarınca yapılır ve Madde 2.1.2 kapsamındaki işler ayrıca düzenlenir."


def test_cap_per_document(built):
    _, _, d = built
    k1 = d["train"][d["train"]["doc_id"] == "TR-K1"]
    assert len(k1) == CAPS["train"]
    assert k1["article_id"].is_monotonic_increasing


def test_no_leakage_or_duplicates(built):
    _, _, d = built
    keys = {s: set(d[s]["text"].map(normalized_key)) for s in SPLITS}
    docs = {s: set(d[s]["doc_id"]) for s in SPLITS}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        assert not keys[a] & keys[b]
        assert not docs[a] & docs[b]
    for s in SPLITS:
        assert d[s]["text"].map(normalized_key).is_unique


def test_text_is_clean(built):
    _, _, d = built
    allt = pd.concat(d.values())["text"]
    for pattern in [r"\[MASK\]", r"\(Değişik", r"tıklayınız", r"\(\d+\)", r"^MADDE", r"^Madde"]:
        assert not allt.str.contains(pattern).any(), pattern
    assert (allt.str.split().str.len() >= 5).all()


def test_side_files(built):
    raw, out, _ = built
    record = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert set(record["raw_files"]) == {f"{s}.jsonl" for s in SPLITS}
    stats = pd.read_csv(out / "stats.csv")
    assert list(stats["step"].unique()) == ["raw", "split", "clean_filter", "dedupe", "cap"]
    assert len(pd.read_csv(out / "manual_check.csv")) == 3


def test_deterministic(tmp_path):
    raw = make_raw(tmp_path)
    for name in ("a", "b"):
        build(make_cfg(raw, tmp_path / name), tmp_path)
    for f in [f"{s}.parquet" for s in SPLITS] + ["stats.csv", "manual_check.csv"]:
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes(), f


def test_out_dir_protection(tmp_path):
    raw = make_raw(tmp_path)
    out = tmp_path / "out"
    build(make_cfg(raw, out), tmp_path)
    build(make_cfg(raw, out), tmp_path)  # same settings: allowed
    with pytest.raises(ConfigMismatchError):
        build(make_cfg(raw, out, train=3), tmp_path)
