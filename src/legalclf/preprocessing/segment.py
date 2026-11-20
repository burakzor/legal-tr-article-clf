"""Split a TurkCanon document into articles (madde)."""
import re
from dataclasses import dataclass

# hyphen, en dash, em dash, figure dash (U+2012), minus sign (U+2212), horizontal bar (U+2015), colon
DASHES = "-–—‒−―:"
PREFIX_KINDS = {
    "GEÇİCİ": "temporary", "Geçici": "temporary", "MUVAKKAT": "temporary", "Muvakkat": "temporary",
    "EK": "additional", "Ek": "additional",
    "MÜKERRER": "repeated", "Mükerrer": "repeated",
}
# "MADDE 1 –", "Madde 12-", "GEÇİCİ MADDE 3 –", "Muvakkat Madde 1 -", "EK MADDE 2-", "Madde 5/A-", "MADDE 2(1) –"
MARKER = re.compile(
    r"(?:(?P<prefix>" + "|".join(PREFIX_KINDS) + r")\s+)?"
    r"(?:MADDE|Madde)\s*(?P<number>\d+)\s*(?:/(?P<suffix>[A-Za-z]))?\s*(?:\(\d+\)\s*)?"
    r"[" + re.escape(DASHES) + r"]"
)


@dataclass(frozen=True)
class RawArticle:
    idx: int  # position of the article in its document, from 0
    kind: str  # normal | temporary | additional | repeated
    number: int
    suffix: str | None  # "A" for "Madde 5/A"
    raw: str  # article text as cut from the document, marker included


def split_document(text: str) -> tuple[str, list[RawArticle]]:
    """Return (header, articles). The header is everything before the first marker.

    Each article runs from its marker to the next marker. A document without any
    marker returns its whole text as header and an empty list.
    """
    matches = list(MARKER.finditer(text))
    if not matches:
        return text, []
    ends = [m.start() for m in matches[1:]] + [len(text)]
    articles = [
        RawArticle(
            idx=i,
            kind=PREFIX_KINDS.get(m.group("prefix"), "normal"),
            number=int(m.group("number")),
            suffix=m.group("suffix"),
            raw=text[m.start():end],
        )
        for i, (m, end) in enumerate(zip(matches, ends))
    ]
    return text[:matches[0].start()], articles
