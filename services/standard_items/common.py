"""Shared constants and string/match helpers for catalogue pricing."""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = (os.getenv("API_BASE_URL") or "").strip().rstrip("/")
STANDARD_ITEMS_PATH = "/api/standard/items"
_FETCH_TIMEOUT_SEC = 20
_MIN_MATCH_SCORE = 40.0
_CATEGORY_CATALOGUE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "exportStandardItemsByCategory.json",
)

_FILLER_WORDS = frozenset(
    {"some", "any", "the", "a", "an", "of", "from", "with", "for", "and", "my", "your", "please"}
)
# Not catalogue items — dimensions / piles / fragments (LLM sometimes lists these alone).
# Leading words in "floor boards", "garden waste" — search the main noun (last word) first.
_DESCRIPTOR_WORDS = frozenset(
    {
        "floor", "wall", "ceiling", "roof", "ground", "garden", "kitchen", "bathroom",
        "bedroom", "loft", "attic", "basement", "internal", "external", "indoor", "outdoor",
        "wooden", "metal", "plastic", "old", "new", "broken", "full", "half", "large", "small",
        "smaller", "green", "general", "domestic", "commercial", "residential",
    }
)

_NON_ITEM_PHRASES = frozenset(
    {
        "pile", "piles", "part", "parts", "piece", "pieces", "bit", "bits",
        "meter", "meters", "metre", "metres", "length", "width", "height",
        "approximately", "approx", "broken", "smaller", "larger", "area",
    }
)
_MATERIAL_WORDS = frozenset(
    {
        "plastic", "wooden", "metal", "steel", "glass", "fabric", "leather",
        "old", "new", "used", "broken", "damaged", "small", "large", "big",
        "full", "half", "double", "single", "internal", "external",
    }
)


def _normalize(value: str) -> str:
    s = unicodedata.normalize("NFD", str(value))
    return s.encode("ascii", "ignore").decode("ascii").lower().strip()


def _float_gbp(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _item_base_gbp(item: dict[str, Any]) -> float | None:
    for key in ("price", "estimatedPrice"):
        val = _float_gbp(item.get(key))
        if val is not None:
            return val
    return None


def _item_gbp(item: dict[str, Any]) -> float | None:
    """Base catalogue price (outside collection) — not inside surcharges."""
    return _item_base_gbp(item)


def _score_match(search: str, item_name: str) -> float:
    s, n = _normalize(search), _normalize(item_name)
    if not s or not n:
        return 0.0
    if s == n:
        return 100.0
    if n.startswith(s) or s in n:
        return 75.0
    if len(s) <= 4:
        return 65.0 if re.search(rf"\b{re.escape(s)}\b", n) else 0.0
    i = 0
    for ch in s:
        j = n.find(ch, i)
        if j < 0:
            return 0.0
        i = j + 1
    return 45.0


def _pick_best(
    search: str,
    items: list[dict[str, Any]],
    *,
    customer_phrase: str = "",
) -> dict[str, Any] | None:
    if not items:
        return None
    customer_words = _customer_words_for_match(customer_phrase) if customer_phrase else []

    def rank(it: dict[str, Any]) -> float:
        name = str(it.get("itemName") or "")
        score = _score_match(search, name)
        n_norm = _normalize(name)
        for w in customer_words:
            if re.search(rf"\b{re.escape(w)}\b", f" {n_norm} "):
                score += 12.0
            elif w.endswith("s") and re.search(rf"\b{re.escape(w[:-1])}\b", f" {n_norm} "):
                score += 12.0
        return score

    if len(items) == 1:
        if rank(items[0]) >= _MIN_MATCH_SCORE:
            return items[0]
        return None
    best = max(items, key=rank)
    if rank(best) < _MIN_MATCH_SCORE:
        return None
    return best


def _core_words(phrase: str) -> list[str]:
    """Words used for API search term (drops filler + material, keeps item nouns)."""
    words: list[str] = []
    for raw in _normalize(phrase).split():
        w = raw.strip("'-")
        if len(w) < 2 or w in _FILLER_WORDS or w in _MATERIAL_WORDS:
            continue
        if w in ("rubbish", "garbage", "trash"):
            continue
        words.append(w)
    return words


def _customer_words_for_match(phrase: str) -> list[str]:
    """Words used to compare customer phrase vs catalogue itemName (stricter)."""
    words: list[str] = []
    for raw in _normalize(phrase).split():
        w = raw.strip("'-")
        if len(w) < 2 or w in _FILLER_WORDS or w in _MATERIAL_WORDS:
            continue
        words.append(w)
    return words


def _singularize(word: str) -> str:
    w = word.lower()
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _catalogue_search_candidates(phrase: str) -> list[str]:
    """
    General API search order for any item phrase (door, screw, board, sofa, …).

    Rule: last word = main item (door, screw, board). Earlier words are usually
    notation (floor, internal, wooden) and are tried only after the main noun.
    """
    low = _normalize(phrase)
    words = _core_words(phrase)
    if not low:
        return []

    seen: set[str] = set()
    out: list[str] = []

    def add(term: str) -> None:
        t = re.sub(r"\s+", " ", (term or "").strip().lower())
        if len(t) >= 2 and t not in seen:
            seen.add(t)
            out.append(t)

    add(low)

    if not words:
        return out

    head = _singularize(words[-1])
    tail = words[-1]

    if len(words) == 1:
        add(head)
        if tail != head:
            add(tail)
        return out

    # modifier(s) + main item — e.g. fire door, floor board, wood screw
    compound = " ".join(words[:-1] + [head])
    add(compound)
    add(compound.replace(" ", ""))

    # Main item first (screw, door, board) before location/material words
    add(head)
    if tail != head:
        add(tail)

    add(" ".join(words))

    for w in words[:-1]:
        if w not in _DESCRIPTOR_WORDS:
            add(w)

    return out


def _main_catalogue_search_term(phrase: str) -> str:
    """Best guess search term for display (main noun when phrase has descriptor + item)."""
    candidates = _catalogue_search_candidates(phrase)
    if not candidates:
        return ""
    words = _core_words(phrase)
    if len(words) >= 2:
        head = _singularize(words[-1])
        if head in candidates:
            return head
    return candidates[0]


def _phrase_matches_catalogue(customer_phrase: str, item_name: str) -> bool:
    """True only when the customer's item description matches the catalogue itemName."""
    cw = _customer_words_for_match(customer_phrase)
    if not cw:
        return False

    n_words = [w for w in re.split(r"[^a-z0-9]+", _normalize(item_name)) if len(w) >= 2]
    c_norm = " ".join(cw)
    n_join = " ".join(n_words)

    if c_norm == n_join:
        return True
    if c_norm in n_join or n_join in c_norm:
        return True

    n_blob = f" {n_join} "

    def _word_in_catalogue(word: str) -> bool:
        if re.search(rf"\b{re.escape(word)}\b", n_blob):
            return True
        if word.endswith("s") and len(word) > 3:
            return bool(re.search(rf"\b{re.escape(word[:-1])}\b", n_blob))
        return bool(re.search(rf"\b{re.escape(_singularize(word))}\b", n_blob))



def _is_collectible_phrase(phrase: str) -> bool:
    low = _normalize(phrase)
    if not low or low in _NON_ITEM_PHRASES:
        return False
    words = _customer_words_for_match(phrase)
    if not words:
        return False
    if len(words) == 1 and words[0] in _NON_ITEM_PHRASES:
        return False
    return True

