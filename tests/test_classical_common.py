import pandas as pd

from legalclf.classical.common import (FULL, doc_bootstrap_ci, macro_f1, truncate_texts,
                                       turkish_lower)


def test_truncate_texts_keeps_first_k_words():
    texts = pd.Series(["bir iki üç dört", "tek"])
    assert truncate_texts(texts, 2).tolist() == ["bir iki", "tek"]
    assert truncate_texts(texts, FULL).tolist() == texts.tolist()


def test_turkish_lower():
    assert turkish_lower("İSTANBUL IRMAK") == "istanbul ırmak"
    assert turkish_lower("İl") == "il"  # str.lower() would give "i̇l" (3 code points)


def test_macro_f1_ignores_classes_absent_from_y_true():
    assert macro_f1(["KANUN", "TEBLIG"], ["KANUN", "TEBLIG"]) == 1.0


def test_doc_bootstrap_ci_is_reproducible_and_contains_point_estimate():
    y_true = ["KANUN"] * 30 + ["TEBLIG"] * 30
    y_pred = ["KANUN"] * 25 + ["TEBLIG"] * 5 + ["TEBLIG"] * 28 + ["KANUN"] * 2
    docs = [f"d{i // 3}" for i in range(60)]
    ci = doc_bootstrap_ci(y_true, y_pred, docs, n_boot=200, seed=0)
    assert ci == doc_bootstrap_ci(y_true, y_pred, docs, n_boot=200, seed=0)
    assert ci[0] <= macro_f1(y_true, y_pred) <= ci[1]
