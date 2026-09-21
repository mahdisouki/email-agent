"""LLM / regex extraction of collectible items from customer text."""
from __future__ import annotations

import json
import re
from typing import Any

from services.standard_items.common import (
    _NON_ITEM_PHRASES,
    _is_collectible_phrase,
    _normalize,
)
from services.standard_items.text_prep import prepare_content_main

def _parse_json_array(raw: str) -> list[Any] | None:
    """
    Parse a JSON array of item objects.

    Treats {} / empty object as empty list (small models often return {} when
    format=json is set). Unwraps {"items": [...]} / {"data": [...]} shapes.
    Returns None only when the payload is unusable prose or invalid JSON.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```\w*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw).strip()

    def _from_data(data: Any) -> list[Any] | None:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if not data:
                return []  # {} → empty, trigger retry
            for key in ("items", "data", "results", "extracted_items", "line_items"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
            if any(k in data for k in ("phrase", "item", "name")):
                return [data]
            return []
        return None

    try:
        data = json.loads(raw)
        out = _from_data(data)
        if out is not None:
            return out
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[[\s\S]*\]", raw)
    if m:
        try:
            data = json.loads(m.group(0))
            out = _from_data(data)
            if out is not None:
                return out
        except json.JSONDecodeError:
            pass

    m_obj = re.search(r"\{[\s\S]*\}", raw)
    if m_obj:
        try:
            data = json.loads(m_obj.group(0))
            out = _from_data(data)
            if out is not None:
                return out
        except json.JSONDecodeError:
            pass
    return None


def llm_extract_items(content_main: str) -> tuple[list[dict[str, Any]], str | None]:
    """
    Ask the LLM what physical items the customer wants collected.

    Returns (items, error). items: [{"phrase": "bags of rubbish", "quantity": 1}, ...]
    Freeform British English — no fixed message format. Quantities come from the message.
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

    system = (
        "You extract collectible waste/furniture items for UK pricing. "
        "Your entire reply MUST be a JSON array starting with [ and ending with ]. "
        "Never reply with {}, never wrap in an object. No markdown, no commentary."
    )

    prompt = f"""From this customer message, list every distinct thing they want collected or disposed of.

Customers write freely — no standard layout. You must read quantities carefully.

Rules:
1. Output ONLY a JSON array. Example shape:
   [{{"phrase": "CRT TV", "quantity": 1}}, {{"phrase": "garden waste bags", "quantity": 17}}]
2. Never return {{}} or {{"items": ...}}. Always a top-level array: [...] or [] if nothing.
3. One row per distinct item group. quantity MUST match the number in the message.
   Examples:
   - "2 1100 litre rubbish bins and 9 smaller bins" →
     [{{"phrase": "1100 litre rubbish bins", "quantity": 2}}, {{"phrase": "smaller bins", "quantity": 9}}]
   - "CRT 26 inch TV, LCD 26 inch TV, spin dryer, 17 bags of garden waste" →
     [{{"phrase": "CRT TV", "quantity": 1}}, {{"phrase": "LCD TV", "quantity": 1}}, {{"phrase": "spin dryer", "quantity": 1}}, {{"phrase": "garden waste bags", "quantity": 17}}]
   - "sofa and a double mattress" →
     [{{"phrase": "sofa", "quantity": 1}}, {{"phrase": "double mattress", "quantity": 1}}]
   - Subject/context is paint and body is "4x 10L 17x 5L 13x 2.5L 11x 1L" →
     [{{"phrase": "10 litre paint cans", "quantity": 4}}, {{"phrase": "5 litre paint cans", "quantity": 17}}, {{"phrase": "2.5 litre paint cans", "quantity": 13}}, {{"phrase": "1 litre paint cans", "quantity": 11}}]
4. Keep useful size/type words (CRT, LCD, 1100 litre, double, garden, paint, etc.).
5. Include TVs, monitors, bins, bags, dryers, furniture, appliances, rubble, paint cans, etc.
6. Do NOT invent items not in the message. If only sizes like "10L" appear but the message is about paint, label them as paint cans of that size.
7. Do NOT include greetings, names, phones, addresses, or questions.
8. If nothing to collect, return [] (empty array).

Customer message:
{text[:6000]}"""

    def _rows_to_items(arr: list[Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for row in arr:
            if not isinstance(row, dict):
                continue
            phrase = str(
                row.get("phrase") or row.get("item") or row.get("name") or ""
            ).strip()
            if not phrase or len(phrase) < 2:
                continue
            try:
                qty = max(1, int(row.get("quantity") or row.get("count") or 1))
            except (TypeError, ValueError):
                qty = 1
            if _is_collectible_phrase(phrase) or _looks_like_waste_item(phrase):
                items.append({"phrase": phrase, "quantity": qty})
        return items

    def _call(
        *,
        temperature: float,
        retry_hint: str = "",
        json_mode: bool = True,
    ) -> tuple[list[dict[str, Any]], str | None, str]:
        user = prompt
        if retry_hint:
            user = (
                f"{prompt}\n\n{retry_hint}\n"
                "Reply with ONLY a JSON array like "
                '[{"phrase":"garden waste bags","quantity":17}].'
            )
        try:
            raw = llm_chat(
                system=system,
                user=user,
                temperature=temperature,
                json_mode=json_mode,
            )
        except Exception as exc:
            return [], f"LLM request failed: {exc}", ""

        arr = _parse_json_array(raw)
        if arr is None:
            return [], f"LLM did not return valid JSON: {(raw or '')[:240]}", (raw or "")
        items = _rows_to_items(arr)
        if not items and arr == []:
            return [], "LLM returned empty array []", (raw or "")
        if not items:
            return [], f"LLM JSON had no usable items: {(raw or '')[:240]}", (raw or "")
        return items, None, (raw or "")

    items, err, raw1 = _call(temperature=0.05)
    if not items:
        object_hint = ""
        if raw1.strip() in ("{}", "{ }") or (raw1.strip().startswith("{") and "[" not in raw1):
            object_hint = (
                "You previously returned a JSON object {}. That is WRONG. "
                "You MUST return a JSON array starting with [ ."
            )
        retry_items, retry_err, raw2 = _call(
            temperature=0.0,
            retry_hint=(
                f"{object_hint} "
                "IMPORTANT: previous answer was empty or invalid. "
                "List every TV, monitor, stand, dryer, bag, bin, sofa, mattress mentioned. "
                "Use the numbers from the message (e.g. 17 bags)."
            ).strip(),
        )
        if retry_items:
            items, err = retry_items, None
        else:
            retry2_items, retry2_err, _ = _call(
                temperature=0.0,
                json_mode=False,
                retry_hint=(
                    "FINAL ATTEMPT. Output nothing except a JSON array. "
                    "Start with [ end with ]. Example: "
                    '[{"phrase":"CRT TV","quantity":1},{"phrase":"garden waste bags","quantity":17}]'
                ),
            )
            if retry2_items:
                items, err = retry2_items, None
            else:
                return [], err or retry_err or retry2_err or "LLM returned no extractable items"

    # Second pass: catch missed numbered groups (e.g. "2 large and 9 smaller bins")
    if items and _message_may_have_multiple_item_groups(text):
        missed, miss_err = _llm_missed_numbered_groups(
            text=text,
            already=items,
            llm_chat=llm_chat,
            system=system,
            rows_to_items=_rows_to_items,
        )
        if missed:
            items = _merge_extracted_item_lists(items, missed)
        elif miss_err and not err:
            # keep primary items; optional note is not fatal
            pass

    return items, err


def _message_may_have_multiple_item_groups(text: str) -> bool:
    """True when the message likely has more than one quantity/item group."""
    low = (text or "").lower()
    qty_nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", low) if 1 <= int(n) <= 500]
    # Drop obvious non-quantities like 1100 in "1100 litre" still counts as size — keep it;
    # trigger when 2+ distinct quantity-ish numbers OR "and" between item phrases.
    if len(qty_nums) >= 2:
        return True
    if re.search(r"\band\b.+\b(bins?|bags?|tvs?|sofas?|mattress|chairs?|doors?)\b", low):
        return True
    return False


def _item_group_stem(phrase: str) -> str:
    low = _normalize(phrase)
    low = re.sub(r"^\d+\s*", "", low)
    for noise in ("litre", "liter", "smaller", "small", "large", "rubbish", "general", "waste"):
        low = re.sub(rf"\b{noise}\b", " ", low)
    words = [w for w in low.split() if w]
    return words[-1] if words else low


def _merge_extracted_item_lists(
    primary: list[dict[str, Any]],
    extras: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append missed groups; if same stem already present, keep the higher quantity."""
    out = [dict(x) for x in primary]
    stems = {_item_group_stem(x["phrase"]): i for i, x in enumerate(out)}
    for extra in extras:
        phrase = str(extra.get("phrase") or "").strip()
        if not phrase:
            continue
        try:
            qty = max(1, int(extra.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1
        stem = _item_group_stem(phrase)
        # Prefer phrase without a leading duplicate count ("2 1100 litre…" → keep descriptive)
        clean_phrase = re.sub(r"^\d+\s+", "", phrase).strip() or phrase
        if stem in stems:
            i = stems[stem]
            # Different type of same noun (1100 litre vs smaller) — stems may collide on "bin"
            # Keep separate if phrases clearly differ beyond the shared noun.
            existing = out[i]["phrase"].lower()
            if clean_phrase.lower() not in existing and existing not in clean_phrase.lower():
                # Distinct variants sharing stem (e.g. 1100 litre bins vs smaller bins)
                out.append({"phrase": clean_phrase, "quantity": qty})
                stems[f"{stem}:{len(out)}"] = len(out) - 1
            else:
                out[i]["quantity"] = max(int(out[i].get("quantity") or 1), qty)
        else:
            out.append({"phrase": clean_phrase, "quantity": qty})
            stems[stem] = len(out) - 1
    return out


def _llm_missed_numbered_groups(
    *,
    text: str,
    already: list[dict[str, Any]],
    llm_chat,
    system: str,
    rows_to_items,
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Second LLM pass: ask whether any numbered item groups were missed
    (e.g. extracted 2×1100L bins but missed 9 smaller bins).
    """
    already_payload = [
        {"phrase": it.get("phrase"), "quantity": it.get("quantity")} for it in already
    ]
    user = f"""Customer message:
{text[:6000]}

Items already extracted:
{json.dumps(already_payload, ensure_ascii=False)}

Task: did we miss any numbered groups in the message?
Look for patterns like "2 X and 9 Y", "17 bags", "3 sofas and a fridge".
Return ONLY a JSON array of MISSING items (not already covered above), each:
{{"phrase": "<item without leading count>", "quantity": <number from the message>}}

Examples of a miss:
- Message: "2 1100 litre rubbish bins and 9 smaller bins"
- Already: [{{"phrase":"2 1100 litre rubbish bins","quantity":2}}]
- Missing: [{{"phrase":"smaller bins","quantity":9}}]

If nothing was missed, return [].
Do not invent items. Do not return {{}}. Array only."""

    try:
        raw = llm_chat(
            system=system
            + " Return ONLY missed items as a JSON array, or [] if complete.",
            user=user,
            temperature=0.0,
            json_mode=True,
        )
    except Exception as exc:
        return [], f"missed-groups pass failed: {exc}"

    arr = _parse_json_array(raw)
    if arr is None:
        # one plain-text retry
        try:
            raw = llm_chat(
                system=system,
                user=user + "\n\nReply with ONLY [...] — start with [.",
                temperature=0.0,
                json_mode=False,
            )
        except Exception as exc:
            return [], f"missed-groups pass failed: {exc}"
        arr = _parse_json_array(raw)
        if arr is None:
            return [], f"missed-groups invalid JSON: {(raw or '')[:200]}"

    missed = rows_to_items(arr)
    return missed, None


def _looks_like_waste_item(phrase: str) -> bool:
    """Accept LLM phrases that mention a common waste noun even if filters are strict."""
    low = _normalize(phrase)
    if not low or len(low) < 3:
        return False
    needles = (
        "bin", "bag", "sofa", "mattress", "fridge", "freezer", "wardrobe",
        "door", "chair", "table", "desk", "rubble", "waste", "rubbish",
        "tile", "board", "appliance", "furniture", "bed", "carpet", "pram",
        "suitcase", "boiler", "sink", "toilet", "bath", "tv", "television",
        "monitor", "crt", "lcd", "dryer", "spin", "garden", "stand",
    )
    return any(n in low for n in needles)


def _message_has_written_item_list(text: str) -> bool:
    """
    Soft gate only: customer wrote an item list / quantities in free text.
    Used so vision detectedItems do not override a successful LLM extraction.
    """
    low = (text or "").lower()
    if len(low) < 20:
        return False
    has_qty = bool(re.search(r"\b\d+\s*[x×]?\s*[a-z]", low))
    has_noun = bool(
        re.search(
            r"\b(bins?|bags?|sofas?|mattress(?:es)?|fridges?|doors?|chairs?|"
            r"tables?|rubbish|waste|rubble|tiles?|wardrobe|furniture|appliance|"
            r"tvs?|televisions?|monitors?|crt|lcd|dryers?|garden)\b",
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
        )
    )
    return bool((has_qty and has_noun) or (written_intent and has_noun))


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


_PAINT_SIZE_QTY_RE = re.compile(
    r"(\d+)\s*[x×]\s*(\d+(?:\.\d+)?)\s*[lL]\b",
    re.I,
)


def _has_paint_context(
    text: str,
    *,
    subject: str = "",
    detected_items: list[dict[str, Any]] | None = None,
) -> bool:
    """True when subject, body, or vision clearly refer to paint."""
    blob = f"{subject}\n{text}".lower()
    if re.search(
        r"\bpaints?\b|"
        r"\bpaint\s*(?:tin|tins|can|cans|disposal|collection|waste)\b|"
        r"\b(?:tin|tins|can|cans)\s+of\s+paint\b",
        blob,
    ):
        return True
    for row in detected_items or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("phrase") or "").lower()
        if "paint" in name:
            return True
    return False


def _extract_paint_size_quantity_groups(text: str) -> list[dict[str, Any]]:
    """
    Parse size-only paint lists: "4x 10L 17x 5L 13x 2.5L 11x 1L"
    → one row per size with phrase "10 litre paint cans", quantity 4, etc.
    """
    items: list[dict[str, Any]] = []
    for m in _PAINT_SIZE_QTY_RE.finditer(text or ""):
        try:
            qty = max(1, int(m.group(1)))
        except (TypeError, ValueError):
            continue
        litres = m.group(2).strip()
        # Normalise "2.50" → keep as given; strip trailing .0
        if re.fullmatch(r"\d+\.0+", litres):
            litres = litres.split(".", 1)[0]
        phrase = f"{litres} litre paint cans"
        items.append({"phrase": phrase, "quantity": qty, "source": "paint_size_list"})
    return items


def extract_items_with_paint_size_override(
    content_main: str,
    *,
    subject: str = "",
    detected_items: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """
    Prefer deterministic paint size groups when context is paint + Nx …L list.
    Otherwise fall through to LLM extraction.

    Returns (items, extraction_method, llm_error).
    """
    text = prepare_content_main(content_main)
    if _has_paint_context(text, subject=subject, detected_items=detected_items):
        sized = _extract_paint_size_quantity_groups(text)
        if sized:
            return sized, "paint_size_list", None

    items, err = llm_extract_items(content_main)
    return items, "llm", err


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


