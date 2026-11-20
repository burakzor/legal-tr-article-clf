from legalclf.preprocessing.filters import is_boilerplate, is_too_short, normalized_key


def test_boilerplate():
    assert is_boilerplate("Bu yayımı tarihinde yürürlüğe girer.", 60)
    assert is_boilerplate("Bu hükümlerini Çevre ve Şehircilik Bakanı yürütür.", 60)
    assert is_boilerplate("Örnek Yönetmeliği yürürlükten kaldırılmıştır.", 60)
    assert not is_boilerplate("Kurul yılda en az dört kez toplanır.", 60)


def test_long_article_with_boilerplate_phrase_is_kept():
    long_text = " ".join(["Kurul kararlarını yazılı olarak bildirir."] * 15) + " Bu madde yayımı tarihinde yürürlüğe girer."
    assert not is_boilerplate(long_text, 60)


def test_too_short():
    assert is_too_short("Mülga", 5)
    assert not is_too_short("Kurul yılda dört kez toplanır.", 5)


def test_normalized_key_ignores_case_spacing_and_digits():
    a = normalized_key("Bu 1/1/2020 tarihinde yürürlüğe girer.")
    assert a == normalized_key("bu  2/3/2021 tarihinde Yürürlüğe girer.")
    assert a != normalized_key("Bu 1/1/2020 tarihinde yürürlükten kalkar.")
