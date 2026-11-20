"""Article filters and the normalized key used for duplicate detection."""
import hashlib
import re

# standard articles that read the same in every legislation type
BOILERPLATE = re.compile(
    r"yürürlüğe girer|yürürlüğe girmiştir|hükümlerini .{0,80}yürütür|yürürlükten kaldırılmıştır|^\s*\(?[Mm]ülga",
    re.S,
)
DIGIT = re.compile(r"\d")


def is_boilerplate(text: str, max_words: int) -> bool:
    return len(text.split()) < max_words and BOILERPLATE.search(text) is not None


def is_too_short(text: str, min_words: int) -> bool:
    return len(text.split()) < min_words


def normalized_key(text: str) -> str:
    """Fingerprint that ignores case, spacing and digits, so dated variants of a text match."""
    norm = DIGIT.sub("0", " ".join(text.lower().split()))
    return hashlib.md5(norm.encode("utf-8")).hexdigest()
