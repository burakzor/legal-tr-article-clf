"""Cleaning rules for a single article.

Each rule is a small function so it can be tested on its own; `clean_article` applies
them in a fixed order. Rules expect whitespace-normalized text.
"""
import re
from dataclasses import dataclass

from legalclf.preprocessing.segment import MARKER

# "Ekleri için tıklayınız" and everything after it (the Resmî Gazete table of the web page)
WEB_ARTIFACT = re.compile(r"\s*(?<!\w)[Ee]k(?:lerini|leri|ler|ini|i)?(?: görmek)? [İi]çin tıklayınız.*\Z", re.S)
_WORD = r"[A-Za-zÇĞİÖŞÜçğıöşü]+"
# "(Değişik:RG-26/9/2024-32674)", "(Ek: 16/5/2012-6307/3 md.)", "(Değişik birinci fıkra:RG-...)",
# "(Başlığı ile Birlikte Değişik:RG-...)"; up to 3 words before and after the keyword. Not "(Ek-1)".
AMENDMENT_NOTE = re.compile(
    rf"\(\s*(?:{_WORD}\s+){{0,3}}(?:[Dd]eğişik|Ek|[Mm]ülga|İptal|Yeniden düzenleme|Değiştirilerek kabul|Aynen kabul)"
    rf"(?:\s+{_WORD}){{0,3}}(?:\s*-\s*R\.\s*G\.)?\s*:[^()]{{0,300}}\)"
)
# section heading left at the end of an article: "... tanımlar. İKİNCİ BÖLÜM Genel Esaslar Başvuru".
# With an upper-case ordinal ("ÜÇÜNCÜ BÖLÜM ...:") everything to the end is heading, colons included.
SECTION_HEADING = re.compile(
    r"(?<=[.:;])\s*(?:[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]*\s+)?(?:KISIM|BÖLÜM|KİTAP|Kısım|Bölüm|Kitap)\b[^.:;]*\Z"
)
SECTION_HEADING_ORDINAL = re.compile(r"(?<=[.:;)])\s*[A-ZÇĞİÖŞÜ]{3,}\s+(?:KISIM|BÖLÜM|KİTAP)\b.*\Z", re.S)
# tail after the last sentence end: the next article's heading ("... belirlemektir. Kapsam")
TAIL = re.compile(r"(?<=[.:;])\s*([^.:;]{1,200})\Z")
TAIL_MAX_WORDS = 12
# "[MASK]" plus a suffix the masking left behind ("[MASK]'nun")
MASK_WITH_SUFFIX = re.compile(r"\[MASK\](?:['’][A-Za-zÇĞİÖŞÜçğıöşü]+)?")
MASK = re.compile(r"\[MASK\]")
# paragraph numbers at the start of the text or after a sentence end; "(13) numaralı" inside a sentence stays
PARAGRAPH_NUMBERS = re.compile(r"(^|[.:;])((?:\s*\(\s*\d+\s*\))+)")
# ... also before a capitalized word ("Hesabı (3) Yukarıda ..."), except references ("(2) Sayılı Cetvel")
PARAGRAPH_BEFORE_CAPITAL = re.compile(r"\(\s*\d+\s*\)\s+(?=(?![Ss]ayılı|SAYILI|[Nn]umaralı|NUMARALI)[A-ZÇĞİÖŞÜ])")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:])")


@dataclass(frozen=True)
class CleaningConfig:
    mask: str = "remove"  # remove | placeholder
    mask_placeholder: str | None = None
    paragraph_numbers: str = "remove"  # remove | keep
    max_words: int = 1000

    def __post_init__(self):
        if self.mask not in ("remove", "placeholder"):
            raise ValueError(f"cleaning.mask must be 'remove' or 'placeholder', got {self.mask!r}")
        if self.mask == "placeholder" and not self.mask_placeholder:
            raise ValueError("cleaning.mask_placeholder is required when cleaning.mask is 'placeholder'")
        if self.paragraph_numbers not in ("remove", "keep"):
            raise ValueError(f"cleaning.paragraph_numbers must be 'remove' or 'keep', got {self.paragraph_numbers!r}")
        if self.max_words < 1:
            raise ValueError("cleaning.max_words must be positive")


def normalize_whitespace(text: str) -> str:
    return SPACE_BEFORE_PUNCT.sub(r"\1", " ".join(text.split()))


def strip_marker(text: str) -> str:
    m = MARKER.match(text)
    return text[m.end():] if m else text


def remove_web_artifacts(text: str) -> str:
    return WEB_ARTIFACT.sub("", text)


def remove_amendment_notes(text: str) -> str:
    return AMENDMENT_NOTE.sub(" ", text)


def remove_trailing_heading(text: str) -> str:
    text = SECTION_HEADING_ORDINAL.sub("", text)
    text = SECTION_HEADING.sub("", text).rstrip()
    m = TAIL.search(text)
    if m and len(m.group(1).split()) <= TAIL_MAX_WORDS and m.group(1)[:1].isupper():
        text = text[:m.start()]
    return text


def handle_mask(text: str, mode: str, placeholder: str | None = None) -> str:
    if mode == "remove":
        return MASK_WITH_SUFFIX.sub(" ", text)
    return MASK.sub(placeholder, text)


def remove_paragraph_numbers(text: str) -> str:
    return PARAGRAPH_BEFORE_CAPITAL.sub(" ", PARAGRAPH_NUMBERS.sub(r"\1 ", text))


def truncate(text: str, max_words: int) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True


def clean_article(raw: str, cfg: CleaningConfig) -> tuple[str, bool]:
    """Return (clean text, truncated)."""
    text = normalize_whitespace(raw)
    text = strip_marker(text)
    text = remove_web_artifacts(text)
    text = remove_amendment_notes(text)
    text = remove_trailing_heading(normalize_whitespace(text))
    text = handle_mask(text, cfg.mask, cfg.mask_placeholder)
    if cfg.paragraph_numbers == "remove":
        text = remove_paragraph_numbers(text)
    return truncate(normalize_whitespace(text), cfg.max_words)
