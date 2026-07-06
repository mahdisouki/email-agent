"""
Standard item pricing for quote replies.

Flow:
  1. contentMain (customer email / form body) → LLM lists items to collect
  2. Each item phrase → GET {API_BASE_URL}/api/standard/items?search=…
  3. Aggregate prices for suggested_replies + pricing metadata
"""
from __future__ import annotations
import json
import os
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = (os.getenv("API_BASE_URL") or "").strip().rstrip("/")
STANDARD_ITEMS_PATH = "/api/standard/items"
_FETCH_TIMEOUT_SEC = 20
_MIN_MATCH_SCORE = 40.0

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
        "green", "general", "domestic", "commercial", "residential",
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


# ---------------------------------------------------------------------------
# contentMain → text for the LLM
# ---------------------------------------------------------------------------


def _comments_from_form(text: str) -> str:
    m = re.search(
        r"Comments:\s*(.+?)(?=\s+Uploaded Items\b|\s+Thank you for using|$)",
        text,
        re.I | re.S,
    )
    return m.group(1).strip() if m else ""


def prepare_content_main(content_main: str) -> str:
    """Use customer wording from contentMain; prefer Comments on quotation forms."""
    text = (content_main or "").strip()
    if not text:
        return ""
    low = text.lower()
    if "first name:" in low and "comments:" in low:
        text = _comments_from_form(text) or text
    return text.strip()


# ---------------------------------------------------------------------------
# LLM: items to price
# ---------------------------------------------------------------------------


def _parse_json_array(raw: str) -> list[Any] | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        m = re.search(r"\[[\s\S]*\]", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            return None


def llm_extract_items(content_main: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Ask the LLM what physical items the customer wants collected.

    Returns (items, error). items: [{"phrase": "bags of rubbish", "quantity": 1}, ...]
    """
    text = prepare_content_main(content_main)
    if not text:
        return [], None

    try:
        from core.llm_service import is_llm_suggest_enabled, llm_chat
    except ImportError:
        return [], "llm_service not available"

    if not is_llm_suggest_enabled():
        return [], "LLM disabled (set OLLAMA_GENERATE_URL and USE_LLM_SUGGEST=true)"

    prompt = f"""You work for a UK waste collection company serving British customers.
Read the customer message and list ONLY the physical items or waste they want collected.

Language: customers write in British English (flat, mobile, postcode, garden, etc.).
Use British item names and spelling in your output.

Rules:
- Include furniture, appliances, rubble, bags of rubbish, doors, mattresses, etc.
- Do NOT include greetings, politeness, verbs (collect, dispose), addresses, or questions.
- Use the main item name — usually the last word (e.g. "door", "screw", "board", "sofa") — not the material/location before it (floor, wooden, internal).
- Not sizes, piles, or "approximately 5 meters". One row per real object only.
- Set quantity from the message (default 1).

Return ONLY a JSON array, no markdown:
[
  {{"phrase": "bags of rubbish", "quantity": 1}},
  {{"phrase": "double mattress", "quantity": 2}}
]

If nothing specific to collect, return [].

Customer message:
{text[:6000]}"""

    try:
        raw = llm_chat(
            system=(
                "You extract waste items for pricing from British customer messages. "
                "Output only valid JSON arrays. Use British English."
            ),
            user=prompt,
            temperature=0.1,
        )
    except Exception as exc:
        return [], f"LLM request failed: {exc}"

    arr = _parse_json_array(raw)
    if arr is None:
        return [], "LLM did not return valid JSON"

    items: list[dict[str, Any]] = []
    for row in arr:
        if not isinstance(row, dict):
            continue
        phrase = str(row.get("phrase") or row.get("item") or row.get("name") or "").strip()
        if not phrase or len(phrase) < 2:
            continue
        try:
            qty = max(1, int(row.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1
        if _is_collectible_phrase(phrase):
            items.append({"phrase": phrase, "quantity": qty})

    return items, None


_REGEX_ITEM_SPECS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d+[- ]?seater\s+sofa\b", re.I), "2 seater sofa"),
    (re.compile(r"\bsofa\s+bed\b", re.I), "sofa bed"),
    (re.compile(r"\bcorner\s+sofa\b", re.I), "corner sofa"),
    (re.compile(r"\bsofa\b", re.I), "sofa"),
    (re.compile(r"\b(?:double|single|king|super\s+king)\s+mattress\b", re.I), "mattress"),
    (re.compile(r"\bmattress\b", re.I), "mattress"),
    (re.compile(r"\bfridge\b|\brefrigerator\b", re.I), "fridge"),
    (re.compile(r"\bfreezer\b", re.I), "freezer"),
    (re.compile(r"\bwashing\s+machine\b", re.I), "washing machine"),
    (re.compile(r"\bwardrobe\b", re.I), "wardrobe"),
    (re.compile(r"\bbed\s+frame\b", re.I), "bed frame"),
    (re.compile(r"\barmchair\b", re.I), "armchair"),
    (re.compile(r"\bchair\b", re.I), "chair"),
]


def _regex_extract_items_fallback(content_main: str) -> list[dict[str, Any]]:
    """Rule-based item list when LLM extraction is empty (common furniture phrases)."""
    text = prepare_content_main(content_main)
    if not text:
        return []
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for pattern, phrase in _REGEX_ITEM_SPECS:
        if pattern.search(text):
            key = phrase.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append({"phrase": phrase, "quantity": 1})
    return items


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = (raw or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def llm_resolve_thread_item_context(
    customer_messages: list[str],
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Read the full customer thread and decide whether items are described,
    merging details from any message (first, middle, or latest — whichever carries the list).

    Returns (result, error). result keys:
      has_item_description: bool
      description: str — consolidated prose for staff/templates
      items: [{"phrase": str, "quantity": int}, ...]
    """
    bodies = [prepare_content_main(m) for m in customer_messages if (m or "").strip()]
    bodies = [b for b in bodies if b]
    if not bodies:
        return None, None

    try:
        from core.llm_service import is_llm_suggest_enabled, llm_chat
    except ImportError:
        return None, "llm_service not available"

    if not is_llm_suggest_enabled():
        return None, "LLM disabled"

    thread_block = "\n\n---\n\n".join(
        f"Customer message {i + 1}:\n{b[:2500]}" for i, b in enumerate(bodies)
    )

    prompt = f"""You work for a UK waste collection company. Read these customer emails from one quote thread (oldest first).

Decide whether the customer has described WHAT physical items need collecting.

Important:
- The item list may appear in the FIRST email, a LATER follow-up, or be spread across several messages — use context, not message position.
- Merge information from all relevant messages into one consolidated description.
- Photos alone, postcode only, or picture clarifications (e.g. "in the middle picture it is the tiles not the BBQ") are NOT a full item list unless they also enumerate what to collect.
- A later message may complete an earlier incomplete thread (e.g. staff asked for a list, customer then lists chairs, bed frame, tiles).

Return ONLY JSON:
{{
  "has_item_description": true,
  "description": "brief consolidated list of what needs collecting",
  "items": [{{"phrase": "dining chair", "quantity": 1}}]
}}

If nothing specific to collect yet:
{{"has_item_description": false, "description": "", "items": []}}

{thread_block[:12000]}"""

    try:
        raw = llm_chat(
            system=(
                "You consolidate waste-collection item descriptions from email threads. "
                "Output only valid JSON. British English."
            ),
            user=prompt,
            temperature=0.1,
            json_mode=True,
        )
    except Exception as exc:
        return None, f"LLM request failed: {exc}"

    parsed = _parse_json_object(raw)
    if not parsed:
        return None, "LLM did not return valid JSON"

    description = str(parsed.get("description") or "").strip()
    items_raw = parsed.get("items")
    items: list[dict[str, Any]] = []
    if isinstance(items_raw, list):
        for row in items_raw:
            if not isinstance(row, dict):
                continue
            phrase = str(row.get("phrase") or row.get("item") or "").strip()
            if not phrase or not _is_collectible_phrase(phrase):
                continue
            try:
                qty = max(1, int(row.get("quantity") or 1))
            except (TypeError, ValueError):
                qty = 1
            items.append({"phrase": phrase, "quantity": qty})

    has_desc = bool(parsed.get("has_item_description"))
    if items and not has_desc:
        has_desc = True
    if has_desc and not description and items:
        description = ", ".join(
            f"{i['quantity']}x {i['phrase']}" if i["quantity"] > 1 else i["phrase"]
            for i in items
        )

    return {
        "has_item_description": has_desc and bool(description or items),
        "description": description[:500],
        "items": items,
    }, None


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


# ---------------------------------------------------------------------------
# API: standard items by itemName search
# ---------------------------------------------------------------------------


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

    return all(_word_in_catalogue(w) for w in cw)


@lru_cache(maxsize=64)
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
    Look up the phrase via the standard-items API.
    standard_item only when an itemName matches the phrase exactly (normalised);
    otherwise custom.
    """
    customer = (phrase or "").strip()
    if not customer or not API_BASE_URL:
        return {"status": "custom"}

    norm_phrase = _normalize(customer)
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
    """Small/generic bin → Household Wheelie Bin; larger by weight → 660L / 1100L."""
    bins = _bin_catalogue_items(items)
    if not bins:
        return None

    weight = _parse_item_weight_kg(detail)
    dims = _parse_dimensions_cm(detail)
    max_dim = max(dims) if dims else None

    def name_low(it: dict[str, Any]) -> str:
        return _normalize(str(it.get("itemName") or ""))

    def household(it: dict[str, Any]) -> bool:
        n = name_low(it)
        return "household" in n and "wheelie" in n

    def litre_660(it: dict[str, Any]) -> bool:
        return "660" in name_low(it)

    def litre_1100(it: dict[str, Any]) -> bool:
        return "1100" in name_low(it)

    small = (weight is not None and weight <= 50) or (max_dim is not None and max_dim <= 100)
    if small:
        for it in bins:
            if household(it):
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
    low = _normalize(phrase)
    if not re.search(r"\bbin\b", low):
        return False
    return not any(spec in low for spec in _BIN_SPECIFIERS)


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
    if matched_via_synonym in ("size_proxy", "household_wheelie_by_size", "generic_bin"):
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
    if row.get("name_match"):
        return False
    if int(row.get("match_count") or 0) <= 1:
        return False
    return len(_customer_words_for_match(phrase)) >= 2


def lookup_item_price_with_synonyms(
    phrase: str,
    *,
    context: str = "",
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Try direct catalogue match, then synonym search via LLM/static map.
    Returns (row, failure_reason).
    """
    detail = _extract_item_detail(phrase, context)

    # Push chair: search by chair (head noun) and use the best catalogue hit — same style as bin.
    if _is_push_chair_phrase(phrase):
        row = lookup_item_price(phrase, context=context)
        return (row, None) if row else (None, "unmatched")

    row = lookup_item_price(phrase, context=context)
    synonyms = llm_catalogue_synonyms(detail)

    if row and _is_weak_catalogue_match(phrase, row):
        row = None

    if not row:
        for synonym in synonyms:
            syn_row = lookup_item_price(synonym, context=context)
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
            syn_row = lookup_item_price(synonym, context=context)
            if syn_row and syn_row.get("name_match"):
                return {
                    **syn_row,
                    "phrase": phrase,
                    "matched_via_synonym": synonym,
                }, None

    return row, None


# ---------------------------------------------------------------------------
# Inside collection surcharges
# ---------------------------------------------------------------------------


def _inside_surcharge_for_line(
    row: dict[str, Any],
    *,
    needs_dismantling: bool,
) -> float:
    if needs_dismantling:
        dismantle = row.get("inside_dismantling_surcharge_gbp")
        if dismantle is not None:
            return float(dismantle)
    inside = row.get("inside_surcharge_gbp")
    return float(inside) if inside is not None else 0.0


def _effective_unit_gbp(
    row: dict[str, Any],
    *,
    collection_side: str | None,
    needs_dismantling: bool,
) -> tuple[float, float]:
    """Return (unit_total, inside_surcharge_applied)."""
    base = float(row.get("base_gbp") if row.get("base_gbp") is not None else row.get("unit_gbp") or 0)
    if (collection_side or "").lower() != "inside":
        return base, 0.0
    surcharge = _inside_surcharge_for_line(row, needs_dismantling=needs_dismantling)
    return base + surcharge, surcharge


def apply_collection_pricing(
    pricing: dict[str, Any] | None,
    *,
    collection_side: str | None = None,
    needs_dismantling: bool = False,
) -> dict[str, Any] | None:
    if not pricing or not pricing.get("line_items"):
        return pricing

    side = (collection_side or "").lower()
    if side != "inside":
        pricing["collection_side"] = collection_side
        pricing["inside_surcharge_applied"] = False
        return pricing

    total = 0.0
    any_estimated = (pricing.get("price_type") or "").lower() == "estimated"
    updated_lines: list[dict[str, Any]] = []

    for it in pricing.get("line_items") or []:
        unit_total, surcharge = _effective_unit_gbp(
            it,
            collection_side=side,
            needs_dismantling=needs_dismantling,
        )
        qty = int(it.get("quantity") or 1)
        line_total = unit_total * qty
        total += line_total
        if it.get("price_type") == "estimated":
            any_estimated = True
        updated_lines.append(
            {
                **it,
                "base_gbp": it.get("base_gbp", it.get("unit_gbp")),
                "inside_surcharge_gbp": surcharge if surcharge else it.get("inside_surcharge_gbp"),
                "inside_dismantling_surcharge_gbp": it.get("inside_dismantling_surcharge_gbp"),
                "unit_gbp": int(unit_total) if unit_total == int(unit_total) else round(unit_total, 2),
                "line_gbp": int(line_total) if line_total == int(line_total) else round(line_total, 2),
                "collection_surcharge_gbp": int(surcharge) if surcharge == int(surcharge) else round(surcharge, 2),
            }
        )

    total_int = int(total) if total == int(total) else round(total, 2)
    return {
        **pricing,
        "ballpark_gbp": total_int,
        "line_items": updated_lines,
        "price_type": "estimated" if any_estimated else pricing.get("price_type"),
        "collection_side": collection_side,
        "inside_surcharge_applied": True,
        "dismantling_surcharge_applied": needs_dismantling,
    }


# ---------------------------------------------------------------------------
# Full pricing from contentMain
# ---------------------------------------------------------------------------


def lookup_prices_for_text(
    content_main: str,
    *,
    collection_side: str | None = None,
    needs_dismantling: bool = False,
) -> dict[str, Any] | None:
    """
    contentMain → LLM items → API price per item → totals + extraction audit trail.
    """
    prepared = prepare_content_main(content_main)
    llm_items, llm_error = llm_extract_items(content_main)
    extraction_method = "llm"
    if not llm_items:
        fallback_items = _regex_extract_items_fallback(content_main)
        if fallback_items:
            llm_items = fallback_items
            extraction_method = "regex_fallback"
            llm_error = None

    extractions: list[dict[str, Any]] = []
    unmatched_items: list[str] = []

    for item in llm_items:
        phrase = item["phrase"]
        qty = int(item.get("quantity") or 1)
        if not _is_collectible_phrase(phrase):
            extractions.append(
                {
                    "phrase": phrase,
                    "quantity": qty,
                    "search_term": _main_catalogue_search_term(phrase),
                    "name_match": False,
                    "matched": False,
                    "skipped": "not_a_collectible_item",
                }
            )
            continue

        row, fail_reason = lookup_item_price_with_synonyms(phrase, context=prepared)
        entry: dict[str, Any] = {
            "phrase": phrase,
            "quantity": qty,
            "search_term": row.get("search") if row else _main_catalogue_search_term(phrase),
            "search_candidates": row.get("search_candidates") if row else _catalogue_search_candidates(phrase)[:6],
            "name_match": row.get("name_match") if row else False,
            "matched": bool(row),
        }
        if row:
            entry["catalogue_match"] = row
            if row.get("matched_via_synonym"):
                entry["matched_via_synonym"] = row["matched_via_synonym"]
        elif fail_reason == "unmatched":
            unmatched_items.append(phrase)
            entry["matched"] = False
            entry["unmatched"] = True
        extractions.append(entry)

    matched = [e for e in extractions if e.get("matched")]
    if not matched:
        base: dict[str, Any] = {
            "ballpark_gbp": None,
            "final_gbp": None,
            "price_type": None,
            "line_items": [],
            "extraction": extractions,
            "extraction_method": extraction_method,
            "content_prepared": prepared[:500] if prepared else "",
            "ambiguous_items": [],
            "unmatched_items": unmatched_items,
        }
        if llm_error or not llm_items:
            base["llm_error"] = llm_error
            return base
        return base

    line_items: list[dict[str, Any]] = []
    total = 0.0
    any_estimated = False
    merged: dict[str, dict[str, Any]] = {}

    for entry in matched:
        row = entry["catalogue_match"]
        qty = int(entry["quantity"])
        item_id = str(row.get("item_id") or entry["phrase"])
        if item_id in merged:
            merged[item_id]["quantity"] += qty
        else:
            merged[item_id] = {**row, "quantity": qty, "phrase": entry["phrase"]}

    for row in merged.values():
        unit = float(row["unit_gbp"])
        qty = int(row["quantity"])
        line_total = unit * qty
        total += line_total
        if row.get("price_type") == "estimated":
            any_estimated = True
        line_items.append(
            {
                "phrase": row.get("phrase"),
                "search": row.get("search"),
                "item_id": row.get("item_id"),
                "item_name": row.get("item_name"),
                "base_gbp": row.get("base_gbp", row.get("unit_gbp")),
                "inside_surcharge_gbp": row.get("inside_surcharge_gbp"),
                "inside_dismantling_surcharge_gbp": row.get("inside_dismantling_surcharge_gbp"),
                "unit_gbp": row.get("unit_gbp"),
                "quantity": qty,
                "line_gbp": int(line_total) if line_total == int(line_total) else round(line_total, 2),
                "price_type": row.get("price_type"),
                "name_match": row.get("name_match"),
                "match_count": row.get("match_count"),
                "match_score": row.get("match_score"),
            }
        )

    total_int = int(total) if total == int(total) else round(total, 2)
    result: dict[str, Any] = {
        "ballpark_gbp": total_int if matched else None,
        "final_gbp": None,
        "price_type": "estimated" if any_estimated or unmatched_items else "exact",
        "line_items": line_items,
        "extraction": extractions,
        "extraction_method": extraction_method,
        "content_prepared": prepared[:500] if prepared else "",
        "llm_error": llm_error,
        "ambiguous_items": [],
        "unmatched_items": unmatched_items,
        "partial_quote": bool(unmatched_items),
    }
    return apply_collection_pricing(
        result,
        collection_side=collection_side,
        needs_dismantling=needs_dismantling,
    )


# ---------------------------------------------------------------------------
# Reply wording helpers
# ---------------------------------------------------------------------------

_PRICE_PLACEHOLDER_RE = re.compile(
    r"£\[PRICE\]\+VAT|£\[PRICE_HALF\]\+VAT|£\[PRICE_LARGE\]\+VAT",
    re.I,
)
_PRICE_PAREN_RE = re.compile(r"(\+VAT)\s*\([^)]*\)", re.I)


def _format_gbp(amount: float | int | str | None) -> str:
    """Whole pounds without decimals (100); pence shown as two decimals (90.50)."""
    if amount is None:
        return "0"
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return "0"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _strip_price_parentheticals(text: str) -> str:
    """Remove notes in parentheses immediately after +VAT."""
    return _PRICE_PAREN_RE.sub(r"\1", text)


def format_pricing_sentence(pricing: dict[str, Any]) -> str:
    ballpark = pricing.get("ballpark_gbp")
    if ballpark is None:
        return ""
    total = _format_gbp(ballpark)
    items: list[dict[str, Any]] = pricing.get("line_items") or []
    estimated = (pricing.get("price_type") or "").lower() == "estimated"

    if estimated or len(items) != 1:
        return f"We estimate this to be £{total}+VAT." if estimated else (
            f"We charge £{total}+VAT to collect and recycle your items."
        )

    name = (items[0].get("item_name") or items[0].get("phrase") or "items").strip()
    return f"We charge £{total}+VAT to collect and recycle your {name}."


def _short_pricing_line(pricing: dict[str, Any]) -> str:
    ballpark = pricing.get("ballpark_gbp")
    if ballpark is None:
        return ""
    total = _format_gbp(ballpark)
    if (pricing.get("price_type") or "").lower() == "estimated":
        return f"We estimate this to be around £{total}+VAT."
    return f"We charge £{total}+VAT to collect and recycle your items."


def apply_pricing_to_suggested_replies(
    replies: list[str],
    pricing: dict[str, Any] | None,
) -> list[str]:
    if not pricing or pricing.get("ballpark_gbp") is None:
        return replies

    ballpark = pricing["ballpark_gbp"]
    formatted_total = _format_gbp(ballpark)
    main_line = format_pricing_sentence(pricing)
    short_line = _short_pricing_line(pricing)
    if not main_line:
        return replies

    ballpark_pat = re.escape(str(ballpark))
    formatted_pat = re.escape(formatted_total)

    out: list[str] = []
    for i, reply in enumerate(replies):
        had_placeholder = bool(_PRICE_PLACEHOLDER_RE.search(reply or ""))
        text = _PRICE_PLACEHOLDER_RE.sub(f"£{formatted_total}+VAT", reply or "")

        # Template already carries the price via £[PRICE]+VAT — do not insert a second line.
        if had_placeholder:
            out.append(_strip_price_parentheticals(text.strip()))
            continue

        if re.search(rf"£\s*(?:{ballpark_pat}|{formatted_pat})", text) and re.search(
            r"\bwe (charge|estimate)\b", text, re.I
        ):
            out.append(_strip_price_parentheticals(text.strip()))
            continue

        if re.search(rf"£\s*(?:{ballpark_pat}|{formatted_pat})\+VAT", text, re.I) and re.search(
            r"\bapproximate price\b", text, re.I
        ):
            out.append(_strip_price_parentheticals(text.strip()))
            continue

        price_line = main_line if i == 0 else (short_line if i % 2 else main_line)
        lines = text.split("\n")
        insert_at = 1
        for j in range(1, len(lines)):
            line = lines[j].strip()
            if not line:
                continue
            low = line.lower()
            if low.startswith("hi ") or low.startswith("dear "):
                continue
            if "thank you" in low or "getting in touch" in low or "getting back" in low:
                insert_at = j + 1
            else:
                insert_at = j
                break
        if insert_at < len(lines) and not lines[insert_at].strip():
            lines[insert_at] = price_line
        else:
            lines.insert(insert_at, "")
            lines.insert(insert_at + 1, price_line)
        out.append(_strip_price_parentheticals("\n".join(lines).strip()))
    return out
