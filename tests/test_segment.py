import pytest

from legalclf.preprocessing.segment import split_document


def markers(text):
    return [(a.kind, a.number, a.suffix) for a in split_document(text)[1]]


@pytest.mark.parametrize("dash", ["-", "–", "—", "‒", "−", "―", ":"])
def test_every_dash_splits(dash):
    text = f"BAŞLIK MADDE 1 {dash} Birinci hüküm. MADDE 2{dash} İkinci hüküm."
    assert markers(text) == [("normal", 1, None), ("normal", 2, None)]


def test_prefixes_give_article_kind():
    text = ("MADDE 1 – A. GEÇİCİ MADDE 1 – B. Geçici Madde 2- C. Muvakkat Madde 3 - D. "
            "EK MADDE 1 – E. Ek Madde 2 – F. Mükerrer Madde 4 – G.")
    assert [k for k, _, _ in markers(text)] == [
        "normal", "temporary", "temporary", "temporary", "additional", "additional", "repeated"]


def test_suffix_and_paragraph_number_after_marker():
    assert markers("MADDE 5/A – Bir hüküm. MADDE 2(1) – (1) Başka hüküm.") == [
        ("normal", 5, "A"), ("normal", 2, None)]


def test_references_are_not_split():
    text = ("MADDE 1 – Bu iş 5 inci madde uyarınca yapılır. Madde 2.1.2 kapsamındaki araçlar "
            "ve Madde 3, 4 ile ilgili hükümler saklıdır.")
    assert markers(text) == [("normal", 1, None)]


def test_header_is_separated_and_articles_run_to_next_marker():
    header, articles = split_document("KAZI HAKKINDA [MASK] Amaç MADDE 1 – (1) Bir. Kapsam MADDE 2 – (1) İki.")
    assert header == "KAZI HAKKINDA [MASK] Amaç "
    assert [a.raw for a in articles] == ["MADDE 1 – (1) Bir. Kapsam ", "MADDE 2 – (1) İki."]
    assert [a.idx for a in articles] == [0, 1]


def test_document_without_marker():
    header, articles = split_document("Ek-1 BAŞVURU FORMU Adı Soyadı")
    assert articles == []
    assert header == "Ek-1 BAŞVURU FORMU Adı Soyadı"
