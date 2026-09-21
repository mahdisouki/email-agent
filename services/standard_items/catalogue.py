"""Catalogue lookup: export JSON first, then live API search."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any
from services.standard_items.extract import _parse_json_array

from services.standard_items.common import (
    API_BASE_URL,
    STANDARD_ITEMS_PATH,
    _CATEGORY_CATALOGUE_PATH,
    _DESCRIPTOR_WORDS,
    _FETCH_TIMEOUT_SEC,
    _FILLER_WORDS,
    _MATERIAL_WORDS,
    _MIN_MATCH_SCORE,
    _catalogue_search_candidates,
    _core_words,
    _customer_words_for_match,
    _float_gbp,
    _item_base_gbp,
    _item_gbp,
    _main_catalogue_search_term,
    _normalize,
    _phrase_matches_catalogue,
    _pick_best,
    _score_match,
    _singularize,
)

def _fetch_items_search(search: str) -> tuple[dict[str, Any], ...]:
    if not API_BASE_URL:
        return ()
    q = urllib.parse.urlencode({"search": search.strip(), "pagination": "false", "limit": "50"})
    url = f"{API_BASE_URL}{STANDARD_ITEMS_PATH}?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return ()
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return ()
    return tuple(i for i in items if isinstance(i, dict))


def classify_item_catalogue_status(phrase: str) -> dict[str, Any]:
    """
    Look up the phrase in exportStandardItemsByCategory.json first, then API.
    standard_item only when an itemName matches the phrase exactly (normalised);
    otherwise custom.
    """
    customer = (phrase or "").strip()
    if not customer:
        return {"status": "custom"}

    norm_phrase = _normalize(customer)
    row = _lookup_exact_in_json(customer)
    if row:
        return {
            "status": "standard_item",
            "item_id": row.get("item_id"),
            "item_name": row.get("item_name"),
        }

    if not API_BASE_URL:
        return {"status": "custom"}

    searches: list[str] = []
    seen: set[str] = set()
    for term in [customer, *_catalogue_search_candidates(customer)]:
        t = term.strip()
        if t and t not in seen:
            seen.add(t)
            searches.append(t)

    for term in searches[:6]:
        for item in _fetch_items_search(term):
            item_name = str(item.get("itemName") or "")
            if _normalize(item_name) == norm_phrase:
                return {
                    "status": "standard_item",
                    "item_id": item.get("_id"),
                    "item_name": item_name,
                }
    return {"status": "custom"}


def enrich_order_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach standard_item | custom status to each extracted item."""
    enriched: list[dict[str, Any]] = []
    for row in items or []:
        if isinstance(row, str):
            row = {"phrase": row, "quantity": 1}
        if not isinstance(row, dict):
            continue
        phrase = str(row.get("phrase") or "").strip()
        if not phrase:
            continue
        try:
            qty = max(1, int(row.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1
        cat = classify_item_catalogue_status(phrase)
        entry: dict[str, Any] = {
            "phrase": phrase,
            "quantity": qty,
            "status": cat["status"],
        }
        if cat["status"] == "standard_item":
            if cat.get("item_id") is not None:
                entry["item_id"] = str(cat["item_id"])
            if cat.get("item_name"):
                entry["item_name"] = cat["item_name"]
        enriched.append(entry)
    return enriched


_BIN_SPECIFIERS = (
    "litre",
    "liter",
    "wheelie",
    "household",
    "1100",
    "660",
    "commercial",
    "rubbish",
    "recycling",
    "garden",
    "litter",
    "storage",
)

_STATIC_CATALOGUE_SYNONYMS: dict[str, list[str]] = {
    "vivarium": ["fish tank", "glass tank", "tank", "display cabinet", "terrarium"],
    "aquarium": ["fish tank", "glass tank", "tank", "display cabinet"],
}

_SIZE_PROXY_SEARCH_TERMS = (
    "fish tank",
    "tank",
    "glass",
    "display cabinet",
    "cabinet",
    "glass table",
)


def _extract_item_detail(phrase: str, context: str) -> str:
    """Pull dimensions/weight from the full message for this item (e.g. Bin: 80 x 38 x 40 cm)."""
    if not context:
        return phrase
    word = phrase.strip().split()[0]
    m = re.search(
        rf"\b{re.escape(word)}s?\s*:?\s*([^\n;]+)",
        context,
        re.I,
    )
    if m:
        detail = m.group(1).strip()
        detail = re.split(
            r"\s+(?=(?:Push|Vivarium|Bin|Sofa|Mattress|Chair)\b)",
            detail,
            maxsplit=1,
            flags=re.I,
        )[0].strip()
        full = f"{word} {detail}".strip()
        if len(full) > len(phrase):
            return full
    return phrase


def _parse_dimensions_cm(text: str) -> tuple[float, float, float] | None:
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*cm",
        text or "",
        re.I,
    )
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _parse_item_weight_kg(text: str) -> float | None:
    m = re.search(
        r"\(\s*(?:around|approx(?:imately)?)?\s*(\d+(?:\.\d+)?)\s*[-–]?\s*(\d+(?:\.\d+)?)?\s*kg",
        text or "",
        re.I,
    )
    if m:
        if m.group(2):
            return (float(m.group(1)) + float(m.group(2))) / 2
        return float(m.group(1))
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?)\b", text or "", re.I)
    return float(m.group(1)) if m else None


def _bin_catalogue_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for it in items:
        name = str(it.get("itemName") or "")
        if "bin" not in _normalize(name):
            continue
        iid = str(it.get("_id") or name)
        if iid in seen:
            continue
        if _item_base_gbp(it) is None:
            continue
        seen.add(iid)
        out.append(it)
    return out


def _pick_generic_bin_item(items: list[dict[str, Any]], detail: str) -> dict[str, Any] | None:
    """Small/generic bin → Household Wheelie Bin / main bin; larger by weight → 660L / 1100L."""
    bins = _bin_catalogue_items(items)
    if not bins:
        return None

    weight = _parse_item_weight_kg(detail)
    dims = _parse_dimensions_cm(detail)
    max_dim = max(dims) if dims else None
    detail_low = _normalize(detail)

    def name_low(it: dict[str, Any]) -> str:
        return _normalize(str(it.get("itemName") or ""))

    def household(it: dict[str, Any]) -> bool:
        n = name_low(it)
        return ("household" in n and "wheelie" in n) or n in ("bin", "bins")

    def any_generic_bin(it: dict[str, Any]) -> bool:
        """Main bin category — not sized commercial 660/1100 litre bins."""
        n = name_low(it)
        if "1100" in n or "660" in n:
            return False
        return "bin" in n

    def litre_660(it: dict[str, Any]) -> bool:
        return "660" in name_low(it)

    def litre_1100(it: dict[str, Any]) -> bool:
        return "1100" in name_low(it)

    # "smaller bins" / "small bins" → household / main bin category
    if _bin_size_hint_is_small(detail_low):
        for it in bins:
            if household(it):
                return it
        for it in bins:
            if any_generic_bin(it):
                return it
        return min(bins, key=lambda x: _item_base_gbp(x) or 9999)

    small = (weight is not None and weight <= 50) or (max_dim is not None and max_dim <= 100)
    if small:
        for it in bins:
            if household(it):
                return it
        for it in bins:
            if any_generic_bin(it):
                return it
        return min(bins, key=lambda x: _item_base_gbp(x) or 9999)

    if weight is not None and weight > 100:
        for it in bins:
            if litre_1100(it):
                return it
    if weight is not None and weight > 50:
        for it in bins:
            if litre_660(it):
                return it

    for it in bins:
        if household(it):
            return it
    for it in bins:
        if any_generic_bin(it):
            return it
    return min(bins, key=lambda x: _item_base_gbp(x) or 9999)


def _is_size_proxy_item(phrase: str) -> bool:
    low = _normalize(phrase)
    return any(
        k in low
        for k in ("vivarium", "aquarium", "terrarium", "reptile tank", "fish tank")
    )


def _find_size_proxy_match(detail: str) -> dict[str, Any] | None:
    """Closest catalogue item by bulk (spaceOfVan) when no direct aquarium/vivarium match."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in _SIZE_PROXY_SEARCH_TERMS:
        for it in _fetch_items_search(term):
            iid = str(it.get("_id") or "")
            if not iid or iid in seen:
                continue
            if _item_base_gbp(it) is None:
                continue
            seen.add(iid)
            candidates.append(it)

    if not candidates:
        return None

    def rank(it: dict[str, Any]) -> float:
        name = _normalize(str(it.get("itemName") or ""))
        space = float(it.get("spaceOfVan") or 0)
        score = space
        if any(w in name for w in ("tank", "glass", "cabinet", "aquarium", "fish")):
            score += 25
        return score

    return max(candidates, key=rank)


def _is_push_chair_phrase(phrase: str) -> bool:
    low = _normalize(phrase)
    return "push" in low and "chair" in low


def _is_generic_bin_phrase(phrase: str) -> bool:
    """True for vague bin wording (bins / smaller bins) without litre/commercial size."""
    low = _normalize(phrase)
    # Match bin and bins (plural) — word-boundary \bbin\b misses "bins"
    if not re.search(r"\bbins?\b", low):
        return False
    return not any(spec in low for spec in _BIN_SPECIFIERS)


def _bin_size_hint_is_small(detail: str) -> bool:
    low = _normalize(detail)
    return bool(
        re.search(r"\b(small|smaller|tiny|little|household|wheelie)\b", low)
    )


def llm_catalogue_synonyms(phrase: str) -> list[str]:
    """Suggest catalogue search terms when the customer phrase has no direct match."""
    phrase = (phrase or "").strip()
    if not phrase:
        return []

    low = _normalize(phrase)
    for key, syns in _STATIC_CATALOGUE_SYNONYMS.items():
        if key in low or low in key:
            return list(syns)

    try:
        from core.llm_service import is_llm_suggest_enabled, llm_chat
    except ImportError:
        return []

    if not is_llm_suggest_enabled():
        return []

    prompt = f"""A UK customer wants this item collected: "{phrase}"

Suggest up to 4 short search terms that might match an item in a British waste company's standard price list.
Use British English (e.g. pushchair, wheelie bin, chest of drawers).
Think of common catalogue names and close synonyms — not materials or sizes.

Return ONLY a JSON array of strings, no markdown:
["reptile tank", "glass tank"]"""

    try:
        raw = llm_chat(
            system=(
                "You map customer item descriptions to UK waste catalogue search terms. "
                "Output only a JSON array of strings."
            ),
            user=prompt,
            temperature=0.1,
        )
    except Exception:
        return []

    arr = _parse_json_array(raw)
    if not arr:
        return []
    out: list[str] = []
    for row in arr:
        term = str(row).strip() if not isinstance(row, dict) else str(
            row.get("term") or row.get("phrase") or row.get("search") or ""
        ).strip()
        if term and len(term) >= 2 and term.lower() != phrase.lower():
            out.append(term)
    return out[:4]


def _row_from_catalogue_item(
    *,
    customer: str,
    chosen: dict[str, Any],
    term: str,
    tried: list[str],
    match_count: int,
    matched_via_synonym: str | None = None,
) -> dict[str, Any]:
    item_name = str(chosen.get("itemName") or "")
    score = _score_match(term, item_name)
    unit = _item_base_gbp(chosen)
    same_name = _phrase_matches_catalogue(customer, item_name)
    price_type = "exact" if match_count == 1 and same_name else "estimated"
    if matched_via_synonym in (
        "size_proxy",
        "household_wheelie_by_size",
        "generic_bin",
        "main_category",
        "export_category",
    ):
        price_type = "estimated"
    row: dict[str, Any] = {
        "phrase": customer,
        "search": term,
        "search_candidates": tried,
        "item_id": chosen.get("_id"),
        "item_name": item_name,
        "name_match": same_name,
        "base_gbp": int(unit) if unit == int(unit) else round(unit, 2),
        "unit_gbp": int(unit) if unit == int(unit) else round(unit, 2),
        "inside_surcharge_gbp": _float_gbp(chosen.get("insidePrice")),
        "inside_dismantling_surcharge_gbp": _float_gbp(
            chosen.get("insideWithDismantlingPrice")
        ),
        "price_type": price_type,
        "match_count": match_count,
        "match_score": round(score, 1),
    }
    if matched_via_synonym:
        row["matched_via_synonym"] = matched_via_synonym
    return row


def lookup_item_price(phrase: str, *, context: str = "") -> dict[str, Any] | None:
    """
    Search catalogue using general term normalisation (singular + head noun).
    Tries a few ordered candidates (e.g. floor boards → floor board → board).
    """
    customer = (phrase or "").strip()
    detail = _extract_item_detail(customer, context)
    if not customer or not API_BASE_URL:
        return None

    if _is_generic_bin_phrase(customer):
        items = list(_fetch_items_search("bin"))
        chosen = _pick_generic_bin_item(items, detail)
        if chosen:
            bins = _bin_catalogue_items(items)
            return _row_from_catalogue_item(
                customer=customer,
                chosen=chosen,
                term="bin",
                tried=["bin"],
                match_count=len(bins),
                matched_via_synonym="household_wheelie_by_size" if _parse_item_weight_kg(detail) else "generic_bin",
            )

    candidates = _catalogue_search_candidates(customer)
    if not candidates:
        return None

    tried: list[str] = []
    for term in candidates[:6]:
        tried.append(term)
        items = list(_fetch_items_search(term))
        if not items:
            continue

        chosen = _pick_best(term, items, customer_phrase=customer)
        if not chosen:
            continue

        item_name = str(chosen.get("itemName") or "")
        score = _score_match(term, item_name)
        unit = _item_base_gbp(chosen)
        if unit is None:
            continue

        same_name = _phrase_matches_catalogue(customer, item_name)
        match_count = len(items)
        return _row_from_catalogue_item(
            customer=customer,
            chosen=chosen,
            term=term,
            tried=tried,
            match_count=match_count,
        )

    return None


def _is_weak_catalogue_match(phrase: str, row: dict[str, Any]) -> bool:
    """Reject generic head-noun matches (e.g. push chair → Dining Chair)."""
    if row.get("matched_via_synonym") in (
        "main_category",
        "export_category",
        "generic_bin",
        "generic_bin_category",
        "household_wheelie_by_size",
        "json_exact",
        "json_fuzzy",
        "json_bin",
    ):
        return False
    if row.get("name_match"):
        return False
    if int(row.get("match_count") or 0) <= 1:
        return False
    return len(_customer_words_for_match(phrase)) >= 2


def _all_json_catalogue_items() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for items in _category_catalogue().values():
        out.extend(items)
    return out


def _lookup_exact_in_json(phrase: str) -> dict[str, Any] | None:
    """Normalised itemName equality against exportStandardItemsByCategory.json."""
    norm = _normalize(phrase)
    if not norm:
        return None
    for it in _all_json_catalogue_items():
        name = str(it.get("itemName") or "")
        if _normalize(name) != norm:
            continue
        if _item_base_gbp(it) is None:
            continue
        return _row_from_catalogue_item(
            customer=phrase,
            chosen=it,
            term=norm,
            tried=[norm],
            match_count=1,
            matched_via_synonym="json_exact",
        )
    return None


def _lookup_fuzzy_in_json(phrase: str) -> dict[str, Any] | None:
    """
    Multi-word phrases only — avoids vague single nouns (e.g. sofa → 3 Seater Sofa).
    """
    words = _customer_words_for_match(phrase)
    if len(words) < 2:
        return None
    candidates = [
        it
        for it in _all_json_catalogue_items()
        if _phrase_matches_catalogue(phrase, str(it.get("itemName") or ""))
        and _item_base_gbp(it) is not None
    ]
    if not candidates:
        return None
    term = _normalize(phrase)
    chosen = _pick_best(term, candidates, customer_phrase=phrase)
    if not chosen:
        return None
    return _row_from_catalogue_item(
        customer=phrase,
        chosen=chosen,
        term=term,
        tried=[term],
        match_count=len(candidates),
        matched_via_synonym="json_fuzzy",
    )


def _lookup_category_in_json(phrase: str) -> dict[str, Any] | None:
    """Map phrase onto an export category key, then pick the best SKU in that category."""
    words = _core_words(phrase)
    if not words:
        return None
    head = _singularize(words[-1])
    if len(head) < 2:
        return None

    resolved = _resolve_export_category(phrase)
    if not resolved:
        return None
    category, items = resolved
    chosen = _pick_main_category_item(head, items, phrase)
    if not chosen:
        chosen = _pick_main_category_item(
            _singularize(_normalize(category).split()[0]), items, phrase
        )
    if not chosen and items:
        unsized = [it for it in items if not re.search(r"\d", str(it.get("itemName") or ""))]
        chosen = min(
            unsized or items,
            key=lambda x: (_item_base_gbp(x) or 9999, len(str(x.get("itemName") or ""))),
        )
    if not chosen:
        return None
    return _row_from_catalogue_item(
        customer=phrase,
        chosen=chosen,
        term=_normalize(category),
        tried=[_normalize(category), head],
        match_count=len(items),
        matched_via_synonym="export_category",
    )


def _lookup_in_json_catalogue(phrase: str, *, context: str = "") -> dict[str, Any] | None:
    """
    Primary catalogue lookup — exportStandardItemsByCategory.json is the source of truth.

    Exact name → generic bin → export category → multi-word fuzzy.
    """
    customer = (phrase or "").strip()
    if not customer or not _category_catalogue():
        return None

    row = _lookup_exact_in_json(customer)
    if row:
        return row

    if _is_generic_bin_phrase(customer):
        bin_items = _category_catalogue().get("Bin") or []
        if bin_items:
            detail = _extract_item_detail(customer, context)
            chosen = _pick_generic_bin_item(bin_items, detail)
            if chosen:
                bins = _bin_catalogue_items(bin_items)
                return _row_from_catalogue_item(
                    customer=customer,
                    chosen=chosen,
                    term="bin",
                    tried=["bin"],
                    match_count=len(bins) or len(bin_items),
                    matched_via_synonym="json_bin",
                )

    row = _lookup_category_in_json(customer)
    if row:
        return row

    return _lookup_fuzzy_in_json(customer)


def _lookup_category_via_api(phrase: str) -> dict[str, Any] | None:
    """API fallback when JSON catalogue has no match."""
    if not API_BASE_URL:
        return None
    words = _core_words(phrase)
    if not words:
        return None
    head = _singularize(words[-1])
    if len(head) < 2:
        return None

    items = list(_fetch_items_search(head))
    if not items and words[-1] != head:
        items = list(_fetch_items_search(words[-1]))
    if not items:
        return None

    chosen = _pick_main_category_item(head, items, phrase)
    if not chosen:
        return None

    related = [
        it
        for it in items
        if re.search(rf"\b{re.escape(head)}s?\b", _normalize(str(it.get("itemName") or "")))
    ]
    return _row_from_catalogue_item(
        customer=phrase,
        chosen=chosen,
        term=head,
        tried=[head],
        match_count=len(related) or len(items),
        matched_via_synonym="main_category",
    )


def _load_category_catalogue() -> dict[str, list[dict[str, Any]]]:
    """Category key → SKUs from exportStandardItemsByCategory.json."""
    try:
        with open(_CATEGORY_CATALOGUE_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for key, items in data.items():
        if not isinstance(key, str) or not isinstance(items, list):
            continue
        priced = [it for it in items if isinstance(it, dict) and _item_base_gbp(it) is not None]
        if priced:
            out[key] = priced
    return out


@lru_cache(maxsize=1)
def _category_catalogue() -> dict[str, list[dict[str, Any]]]:
    return _load_category_catalogue()


def _category_tokens(category: str) -> list[str]:
    parts = re.split(r"[\s/]+|\bor\b", _normalize(category))
    tokens: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) < 2:
            continue
        tokens.append(p)
        sing = _singularize(p)
        if sing != p:
            tokens.append(sing)
    return tokens


def _resolve_export_category(phrase: str) -> tuple[str, list[dict[str, Any]]] | None:
    """
    Map a customer phrase onto an export category key, then its SKUs.

    "smaller bins" → Bin
    "CRT 26 inch TV" → TV
    "17 bags of garden waste" → Waste
    """
    catalogue = _category_catalogue()
    if not catalogue:
        return None

    phrase_low = _normalize(phrase)
    if not phrase_low:
        return None
    phrase_words = set(_core_words(phrase)) | {
        _singularize(w) for w in _core_words(phrase)
    }
    head = _singularize(_core_words(phrase)[-1]) if _core_words(phrase) else ""

    scored: list[tuple[float, str]] = []
    for category, items in catalogue.items():
        cat_n = _normalize(category)
        tokens = _category_tokens(category)
        if not tokens:
            continue
        cat_head = _singularize(tokens[0])
        score = 0.0

        if head and (head == cat_n or head == cat_head or head in tokens):
            score += 100
        if re.search(rf"\b{re.escape(cat_n)}\b", phrase_low):
            score += 80
        for tok in tokens:
            if tok in phrase_words or re.search(rf"\b{re.escape(tok)}s?\b", phrase_low):
                score += 45
                break

        # SKU names in this category share the phrase head noun
        if head:
            for it in items:
                name_n = _normalize(str(it.get("itemName") or ""))
                if re.search(rf"\b{re.escape(head)}s?\b", name_n):
                    score += 30
                    break

        # Vague size words should still land on a real category (smaller bins → Bin)
        if score >= 100:
            scored.append((score, category))
        elif score >= 45 and len(cat_n) >= 3:
            scored.append((score, category))

    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    best_cat = scored[0][1]
    return best_cat, catalogue[best_cat]


def _pick_main_category_item(
    head: str,
    items: list[dict[str, Any]],
    phrase: str,
) -> dict[str, Any] | None:
    """
    Pick the catalogue item that best represents the main category (head noun).

    e.g. "smaller bins" → head "bin" → prefer "Bin" / "Household Wheelie Bin"
    over "1100 Litre Bin" when the phrase has no litre size.
    """
    head_n = _singularize(_normalize(head))
    if not head_n or not items:
        return None
    phrase_low = _normalize(phrase)
    phrase_has_digits = bool(re.search(r"\d", phrase_low))

    scored: list[tuple[float, dict[str, Any]]] = []
    for it in items:
        name = str(it.get("itemName") or "")
        n = _normalize(name)
        if not n or _item_base_gbp(it) is None:
            continue
        if not re.search(rf"\b{re.escape(head_n)}s?\b", n):
            continue

        name_words = [w for w in re.split(r"[^a-z0-9]+", n) if w]
        score = 0.0
        # Exact main category name: "Bin", "Mattress", "Chair"
        if n == head_n or n == f"{head_n}s":
            # Strong only when customer gave no size/number; otherwise prefer sized SKUs
            score += 25 if phrase_has_digits else 100
        if name_words and _singularize(name_words[-1]) == head_n:
            score += 40
        if len(name_words) == 1:
            score += 20 if phrase_has_digits else 35
        elif len(name_words) == 2:
            score += 15
        # Soft generic catalogue labels
        if any(w in n for w in ("household", "general", "standard", "wheelie")):
            score += 20 if not phrase_has_digits else 5
        # Customer gave no size → avoid sized catalogue variants (1100L, 660L, …)
        if not phrase_has_digits and re.search(r"\d", n):
            score -= 60
        # Customer gave a size → strongly favour matching digits in catalogue name
        if phrase_has_digits:
            for dig in re.findall(r"\d+", phrase_low):
                # Ignore tiny counts like "2" when matching catalogue SKUs (1100, 660, …)
                if len(dig) >= 3 and dig in n:
                    score += 90
                elif len(dig) >= 2 and dig in n:
                    score += 40
        score += _score_match(head_n, n) * 0.15
        scored.append((score, it))

    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(str(x[1].get("itemName") or "")), _item_base_gbp(x[1]) or 9999))
    return scored[0][1]


def _lookup_catalogue_row(phrase: str, *, context: str = "") -> dict[str, Any] | None:
    """JSON first, then API search, then API category fallback."""
    row = _lookup_in_json_catalogue(phrase, context=context)
    if row and not _is_weak_catalogue_match(phrase, row):
        return row

    row = lookup_item_price(phrase, context=context)
    if row and not _is_weak_catalogue_match(phrase, row):
        return row

    return _lookup_category_via_api(phrase)


def lookup_item_price_with_synonyms(
    phrase: str,
    *,
    context: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Price lookup: exportStandardItemsByCategory.json first, API fallback.
    Returns (row, failure_reason).
    """
    detail = _extract_item_detail(phrase, context)
    synonyms = llm_catalogue_synonyms(detail)

    row = _lookup_catalogue_row(phrase, context=context)

    if not row:
        for synonym in synonyms:
            syn_row = _lookup_catalogue_row(synonym, context=context)
            if syn_row and not _is_weak_catalogue_match(phrase, syn_row):
                return {
                    **syn_row,
                    "phrase": phrase,
                    "matched_via_synonym": synonym,
                }, None

        if _is_size_proxy_item(detail):
            proxy = _find_size_proxy_match(detail)
            if proxy:
                return _row_from_catalogue_item(
                    customer=phrase,
                    chosen=proxy,
                    term="size_proxy",
                    tried=list(_SIZE_PROXY_SEARCH_TERMS),
                    match_count=1,
                    matched_via_synonym="size_proxy",
                ), None

        return None, "unmatched"

    if not row.get("name_match") and synonyms:
        for synonym in synonyms:
            syn_row = _lookup_in_json_catalogue(synonym, context=context)
            if syn_row and syn_row.get("name_match"):
                return {
                    **syn_row,
                    "phrase": phrase,
                    "matched_via_synonym": synonym,
                }, None
            syn_row = lookup_item_price(synonym, context=context)
            if syn_row and syn_row.get("name_match"):
                return {
                    **syn_row,
                    "phrase": phrase,
                    "matched_via_synonym": synonym,
                }, None

    return row, None


