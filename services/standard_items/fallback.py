"""Fallbacks when LLM item extraction returns nothing."""
from __future__ import annotations

import re
from typing import Any

from services.standard_items.text_prep import prepare_content_main


def message_has_written_item_list(text: str) -> bool:
    """
    True when the customer wrote an item list / quantities in free text.

    Used so vision detectedItems do not override a successful text extraction —
    not to block fallbacks when the LLM returns [].
    """
    low = (text or "").lower()
    if len(low) < 12:
        return False
    has_qty = bool(re.search(r"\b\d+\s*[x×]\s*[a-z]", low))
    has_noun = bool(
        re.search(
            r"\b(bins?|bags?|sofas?|mattress(?:es)?|fridges?|doors?|chairs?|"
            r"tables?|rubbish|waste|rubble|tiles?|wardrobe|furniture|appliance|"
            r"tvs?|televisions?|monitors?|crt|lcd|dryers?|garden|cardboard|"
            r"boxes?|gates?|trellis(?:es)?|fence)\b",
            low,
        )
    )
    written_intent = any(
        x in low
        for x in (
            "we have",
            "i have",
            "these items",
            "need to dispose",
            "need collecting",
            "to be collected",
            "to collect",
            "want rid",
            "quote for",
            "dispose of",
            "would like to be collected",
            "please quote",
        )
    )
    freeform_list = bool(
        has_noun
        and (
            ("," in low and " and " in low)
            or re.search(r"\b\w+\s+and\s+(?:an?\s+)?\w+", low)
        )
    )
    return bool((has_qty and has_noun) or (written_intent and has_noun) or freeform_list)


# Narrow furniture safety net — not a full catalogue mirror
_REGEX_ITEM_SPECS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d+[- ]?seater\s+sofa\b", re.I), "2 seater sofa"),
    (re.compile(r"\bsofa\s+bed\b", re.I), "sofa bed"),
    (re.compile(r"\bcorner\s+sofa\b", re.I), "corner sofa"),
    (re.compile(r"\bsofa\b", re.I), "sofa"),
    (re.compile(r"\b(?:double|single|king|super\s+king)\s+mattress\b", re.I), "double mattress"),
    (re.compile(r"\bmattress\b", re.I), "mattress"),
    (re.compile(r"\bfridge\b|\brefrigerator\b", re.I), "fridge"),
    (re.compile(r"\bfreezer\b", re.I), "freezer"),
    (re.compile(r"\bwashing\s+machine\b", re.I), "washing machine"),
    (re.compile(r"\bwardrobe\b", re.I), "wardrobe"),
    (re.compile(r"\bbed\s+frame\b", re.I), "bed frame"),
    (re.compile(r"\barmchair\b", re.I), "armchair"),
    (re.compile(r"\boffice\s+chairs?\b", re.I), "office chairs"),
    (re.compile(r"\bchair\b", re.I), "chair"),
    (re.compile(r"\bcardboard\s+boxes?\b", re.I), "cardboard boxes"),
]


_FILLER_PREFIX = re.compile(
    r"^(?:an?\s+|the\s+|some\s+|old\s+|an\s+old\s+|a\s+few\s+)+",
    re.I,
)
_LEADING_QTY = re.compile(r"^(\d+)\s*[x×]\s*", re.I)

# Pure numeric sizes: 140 x 190 x 25, 140 x 140
_NUMERIC_DIMS = re.compile(
    r"\b\d+(?:\s*[x×]\s*\d+){1,3}\b",
    re.I,
)
# Letter-prefixed sizes: D550 x H 600, W90 x D60
_LABELLED_DIMS = re.compile(
    r"\b[A-Za-z]\s*\d+\s*[x×]\s*[A-Za-z]?\s*\d+(?:\s*[x×]\s*[A-Za-z]?\s*\d+)*\b",
    re.I,
)
# "1 x Double mattress" — qty + letter (item), not "140 x 190" (digit)
_NX_ITEM_START = re.compile(
    r"(\d{1,3})\s*[x×]\s+(?=[A-Za-z])",
    re.I,
)
_QUOTE_PREFIX = re.compile(
    r"^(?:please\s+)?quote\s+for\s*:?\s*",
    re.I,
)


def _strip_dimensions(text: str) -> str:
    """Remove size specs so they are not mistaken for qty×item markers."""
    out = _NUMERIC_DIMS.sub(" ", text or "")
    out = _LABELLED_DIMS.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


def _clean_item_phrase(phrase: str) -> str:
    phrase = (phrase or "").strip(" .;:-")
    phrase = _FILLER_PREFIX.sub("", phrase).strip()
    # Drop trailing dismantle/assembly notes
    phrase = re.split(
        r"\bdisassembled\b|\binto\s+top\b|\bframe,?\s+and\b",
        phrase,
        maxsplit=1,
        flags=re.I,
    )[0].strip(" .;:-")
    # Collapse repeated words: "table, timber, timber" → keep useful head
    phrase = re.sub(r"\s*,\s*", " ", phrase)
    phrase = re.sub(r"\b(\w+)(?:\s+\1)+\b", r"\1", phrase, flags=re.I)
    return re.sub(r"\s+", " ", phrase).strip()


def _split_nx_item_list(text: str) -> list[dict[str, Any]]:
    """
    Split lists like: 1 x Double mattress … 2 x office chairs …
    Ignores size dims (140 x 190 x 25, D550 x H 600).
    """
    cleaned = _strip_dimensions(text)
    cleaned = _QUOTE_PREFIX.sub("", cleaned).strip()
    if not cleaned:
        return []

    starts = list(_NX_ITEM_START.finditer(cleaned))
    if len(starts) < 1:
        return []

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, m in enumerate(starts):
        try:
            qty = max(1, int(m.group(1)))
        except (TypeError, ValueError):
            qty = 1
        end = starts[i + 1].start() if i + 1 < len(starts) else len(cleaned)
        phrase = _clean_item_phrase(cleaned[m.end() : end])
        if len(phrase) < 3:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"phrase": phrase, "quantity": qty, "source": "nx_fallback"})
    return items


def _split_comma_and_item_list(text: str) -> list[dict[str, Any]]:
    """
    Split human lists: "A, B and C" / "A and B" / "A, B, C"
    Quantities like "1 x" are optional.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    low = raw.lower()
    if "," not in low and " and " not in low:
        return []

    chunk = re.sub(r"\s+plus\s+", " and ", raw, flags=re.I)
    parts = re.split(r"\s*,\s*|\s+and\s+", chunk, flags=re.I)

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for part in parts:
        phrase = (part or "").strip(" .;:-")
        if not phrase:
            continue
        qty = 1
        m = _LEADING_QTY.match(phrase)
        if m:
            try:
                qty = max(1, int(m.group(1)))
            except (TypeError, ValueError):
                qty = 1
            phrase = phrase[m.end() :].strip()
        phrase = _clean_item_phrase(phrase)
        if len(phrase) < 3:
            continue
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append({"phrase": phrase, "quantity": qty, "source": "freeform_fallback"})
    if len(items) < 2:
        return []
    return items


def _regex_furniture_fallback(text: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for pattern, phrase in _REGEX_ITEM_SPECS:
        if pattern.search(text):
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append({"phrase": phrase, "quantity": 1, "source": "regex_fallback"})
    return items


def extract_items_fallback(content_main: str) -> list[dict[str, Any]]:
    """
    When LLM extraction is empty:
    1) N× item lists (prefer over commas — sizes stripped)
    2) comma / and freeform lists
    3) narrow furniture regex
    """
    text = prepare_content_main(content_main)
    if not text:
        return []

    nx_items = _split_nx_item_list(text)
    if len(nx_items) >= 1:
        return nx_items

    freeform = _split_comma_and_item_list(text)
    if freeform:
        return freeform

    return _regex_furniture_fallback(text)
