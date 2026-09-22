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

    from services.standard_items.common import _catalogue_category_names

    category_hint = ", ".join(_catalogue_category_names()[:80])
    category_block = (
        f"\nCommon catalogue categories (for grounding only — still use the customer's words): "
        f"{category_hint}\n"
        if category_hint
        else ""
    )

    system = (
        "You extract collectible waste/furniture/appliance items for UK pricing. "
        "Customers write like humans — full sentences, commas, 'and', or spaces. "
        "A single item in a sentence still counts. Quantities like '1 x' are optional. "
        "Your entire reply MUST be a JSON array starting with [ and ending with ]. "
        "Never reply with {}, never wrap in an object. No markdown, no commentary."
    )

    prompt = f"""From this customer message, list every distinct thing they want collected or disposed of.

Customers write freely. Quantities (1 x, 2x) are OPTIONAL — default quantity 1.
{category_block}
Rules:
1. Output ONLY a JSON array:
   [{{"phrase": "CRT TV", "quantity": 1}}, {{"phrase": "garden waste bags", "quantity": 17}}]
2. Never return {{}} or {{"items": ...}}. Always a top-level array: [...] .
3. One row per distinct physical item. If no number is given, use quantity 1.
4. Single-sentence requests count. Examples:
   - "I wish to dispose of an electric cooker" →
     [{{"phrase": "electric cooker", "quantity": 1}}]
   - "Please remove my old fridge" →
     [{{"phrase": "fridge", "quantity": 1}}]
   - "Need a sofa collected" →
     [{{"phrase": "sofa", "quantity": 1}}]
5. Split lists whatever the punctuation:
   - "Cardboard boxes, garden trellises and an old garden gate"
   - "cardboard boxes garden trellises garden gate"
   - "1 x Double mattress 2 x office chairs"
6. Keep useful type words (electric, double, garden, cardboard, CRT, …).
   Use the customer's everyday words — do NOT invent exact catalogue SKU names.
7. Include appliances (cooker, oven, fridge, washing machine), furniture, garden items,
   cardboard, gates, TVs, bins, bags, rubble, paint, etc.
8. Do NOT invent items not in the message.
9. Do NOT include greetings, names, phones, addresses, form labels, or questions.
10. Return [] ONLY if there is truly nothing physical to collect.
    If the message mentions any object to remove/dispose/collect/quote, you MUST list it.

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
        user_override: str | None = None,
    ) -> tuple[list[dict[str, Any]], str | None, str]:
        user = user_override or prompt
        if retry_hint and not user_override:
            user = (
                f"{prompt}\n\n{retry_hint}\n"
                "Reply with ONLY a JSON array like "
                '[{"phrase":"electric cooker","quantity":1}].'
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
        retry_items, retry_err, _ = _call(
            temperature=0.0,
            retry_hint=(
                f"{object_hint} "
                "IMPORTANT: previous answer was empty or invalid. "
                "The message almost certainly names something to collect. "
                "Even ONE item in a full sentence counts "
                '(e.g. "I wish to dispose of an electric cooker" → '
                '[{"phrase":"electric cooker","quantity":1}]). '
                "Split every distinct collectible thing into its own row. "
                "If no number is written, use quantity 1."
            ).strip(),
        )
        if retry_items:
            items, err = retry_items, None
        else:
            # Focused recovery: short prompt, hard requirement not to return []
            recovery = (
                "Extract collectible items from this UK waste-collection message.\n"
                "Return ONLY a JSON array of {\"phrase\",\"quantity\"}.\n"
                "Single sentences count. Example: "
                '"I wish to dispose of an electric cooker" → '
                '[{"phrase":"electric cooker","quantity":1}]\n'
                "Do NOT return []. If anything physical is mentioned, list it.\n"
                f"{category_block}\n"
                f"Message:\n{text[:4000]}"
            )
            retry2_items, retry2_err, _ = _call(
                temperature=0.0,
                json_mode=True,
                user_override=recovery,
            )
            if not retry2_items:
                retry2_items, retry2_err, _ = _call(
                    temperature=0.0,
                    json_mode=False,
                    user_override=(
                        recovery
                        + "\n\nFINAL: start with [ end with ]. "
                        'Example: [{"phrase":"electric cooker","quantity":1}]'
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
    if re.search(
        r"\band\b.+\b(bins?|bags?|tvs?|sofas?|mattress|chairs?|doors?|"
        r"boxes?|gates?|trellis(?:es)?|tables?|furniture|cardboard|garden)\b",
        low,
    ):
        return True
    # Comma-separated item list without quantities
    if "," in low and re.search(
        r"\b(boxes?|gates?|trellis|sofa|mattress|chair|table|bin|bag|"
        r"cardboard|garden|door|fridge|wardrobe)\b",
        low,
    ):
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
        "cooker", "oven", "hob", "washer", "dishwasher", "microwave",
        "cardboard", "gate", "trellis", "fence", "paint",
    )
    return any(n in low for n in needles)


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


