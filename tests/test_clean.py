import pytest

from legalclf.preprocessing.clean import (
    CleaningConfig, clean_article, handle_mask, normalize_whitespace, remove_amendment_notes,
    remove_paragraph_numbers, remove_trailing_heading, remove_web_artifacts, strip_marker, truncate,
)


def n(text):
    return normalize_whitespace(text)


def test_strip_marker():
    assert strip_marker("MADDE 3 – (1) Kurul toplanır.") == " (1) Kurul toplanır."
    assert n(strip_marker("Geçici Madde 2- Eski hükümler uygulanır.")) == "Eski hükümler uygulanır."


def test_web_artifact_and_everything_after_it_are_removed():
    text = "Bakan yürütür. Ekleri için tıklayınız [MASK] Yayımlandığı Resmî Gazete’nin Tarihi Sayısı 22/4/2020"
    assert remove_web_artifacts(text) == "Bakan yürütür."
    for variant in ["Eki için tıklayınız", "Ekler için tıklayınız", "ekleri için tıklayınız", "Ekleri İçin tıklayınız",
                    "eklerini görmek için tıklayınız Yayımlandığı Resmî Gazete’nin Tarihi Sayısı 15/2/2008 26788"]:
        assert remove_web_artifacts(f"Bakan yürütür. {variant}") == "Bakan yürütür."
    other = "Tabloyu görmek için tıklayınız"
    assert remove_web_artifacts(other) == other


@pytest.mark.parametrize("note", [
    "(Değişik:RG-26/9/2024-32674)",
    "(Ek: 16/5/2012-6307/3 md.)",
    "(Mülga ibare:RG-23/2/2017-29988)",
    "(Ek fıkra:17/1/2019-7161/10 md.)",
    "(Değiştirilerek kabul: 1/2/2018- 7078/34 md.)",
    "(Değişik birinci fıkra:RG-26/9/2024-32674)",
    "(Mülga ikinci fıkra: 2/7/2018 – KHK-703/69 md.)",
    "(Başlığı ile Birlikte Değişik:RG-2/4/2024-32508)",
    "(Başlığı ile birlikte değişik:RG-21/5/2014-29006)",
    "( Değişik ibare:RG-1/1/2015-29222)",
    "(Değişik - R.G.: 12/3/2005-25753)",
])
def test_amendment_notes_are_removed(note):
    assert n(remove_amendment_notes(f"{note}(1) Kurul toplanır.")) == "(1) Kurul toplanır."


def test_annex_reference_is_not_an_amendment_note():
    assert remove_amendment_notes("Form (Ek-1) doldurulur.") == "Form (Ek-1) doldurulur."


@pytest.mark.parametrize("text, expected", [
    ("Usul ve esasları belirlemektir. Kapsam", "Usul ve esasları belirlemektir."),
    ("Yayımı tarihinde yürürlüğe girer. Yürürlükten kaldırılan [MASK]", "Yayımı tarihinde yürürlüğe girer."),
    ("Terimler tanımlanmıştır. İKİNCİ BÖLÜM Genel Esaslar Başvuru", "Terimler tanımlanmıştır."),
    ("Hükümler uygulanır. ÜÇÜNCÜ BÖLÜM Verilecek Giyim Eşyası ve Kullanma Süresi:", "Hükümler uygulanır."),
    ("Yerine işlenmiştir.) İKİNCİ BÖLÜM: Disiplin Esasları", "Yerine işlenmiştir.)"),
    ("Hesaba kaydedilir. Mahalli idareler stajyer kontrolörlüğü giriş sınavı şartları",
     "Hesaba kaydedilir."),
])
def test_trailing_heading_is_removed(text, expected):
    assert remove_trailing_heading(text) == expected


@pytest.mark.parametrize("text", [
    "Kurul toplanır. Kurul kararlarını üyelerinin salt çoğunluğu ile alır ve en geç on beş gün içinde yazılı olarak bildirir",
    "Kurul kararlarını üyelerinin salt çoğunluğu ile alır",
    "Kurul toplanır. kararlar yazılır",
])
def test_real_text_is_not_taken_for_a_heading(text):
    assert remove_trailing_heading(text) == text


def test_mask_remove():
    assert n(handle_mask("Bu [MASK] amacı", "remove")) == "Bu amacı"
    assert n(handle_mask("Kooperatifler [MASK]'nun 80 inci maddesi", "remove")) == "Kooperatifler 80 inci maddesi"
    assert n(handle_mask("Bu [MASK], 5510 sayılı", "remove")) == "Bu, 5510 sayılı"


def test_mask_placeholder():
    assert handle_mask("Bu [MASK] amacı", "placeholder", "XXX") == "Bu XXX amacı"


def test_paragraph_numbers():
    assert n(remove_paragraph_numbers("(1) Kurul toplanır. (2) Kararlar yazılır.")) == "Kurul toplanır. Kararlar yazılır."
    assert n(remove_paragraph_numbers("( 1) Kurul toplanır.")) == "Kurul toplanır."
    assert n(remove_paragraph_numbers("Borçlar Hesabı (3) Yukarıda belirtilen hesaplar")) == \
        "Borçlar Hesabı Yukarıda belirtilen hesaplar"
    for kept in ["Birinci fıkrasının (13) numaralı bendi uygulanır.", "Ekli (1) sayılı cetvelde yer alır.",
                 "Kadrolar (2) SAYILI CETVEL ile belirlenir.", "Bu (1) Sayılı Cetvel uygulanır."]:
        assert remove_paragraph_numbers(kept) == kept


def test_truncate():
    assert truncate("bir iki üç", 5) == ("bir iki üç", False)
    assert truncate("bir iki üç dört", 2) == ("bir iki", True)


def test_clean_article_default():
    raw = "MADDE 3 – (Değişik:RG-1/1/2020-31000) (1) Bu [MASK] amacı; usul ve  esasları belirlemektir. Kapsam "
    assert clean_article(raw, CleaningConfig()) == ("Bu amacı; usul ve esasları belirlemektir.", False)


def test_clean_article_options():
    raw = "MADDE 1 – (1) Bu [MASK] amacı; esasları belirlemektir."
    cfg = CleaningConfig(mask="placeholder", mask_placeholder="XXX", paragraph_numbers="keep")
    assert clean_article(raw, cfg) == ("(1) Bu XXX amacı; esasları belirlemektir.", False)


def test_invalid_config():
    with pytest.raises(ValueError):
        CleaningConfig(mask="drop")
    with pytest.raises(ValueError):
        CleaningConfig(mask="placeholder")
    with pytest.raises(ValueError):
        CleaningConfig(paragraph_numbers="maybe")
