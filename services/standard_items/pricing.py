"""Aggregate item prices and build quote items payload."""
from __future__ import annotations

import re
from typing import Any

from services.standard_items.catalogue import lookup_item_price_with_synonyms
from services.standard_items.common import (
    _catalogue_search_candidates,
    _is_collectible_phrase,
    _main_catalogue_search_term,
)
from services.standard_items.extract import extract_items_with_paint_size_override
from services.standard_items.fallback import (
    extract_items_fallback,
    message_has_written_item_list,
)
from services.standard_items.text_prep import prepare_content_main

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


_DETECTED_STOPWORDS = frozenset(
    {
        "of", "the", "a", "an", "and", "with", "for", "or", "to",
        "litre", "liter", "small", "smaller", "large", "mixed", "general",
        "waste", "rubbish", "heavy", "item", "items",
    }
)


def _item_stem(phrase: str) -> str:
    """Last meaningful word, lightly singularised — for matching 'bins' ↔ 'Bin'."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", (phrase or "").lower())
    words = [w for w in cleaned.split() if w and w not in _DETECTED_STOPWORDS]
    if not words:
        words = cleaned.split() or [""]
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def _normalize_detected_items(
    detected_items: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """[{name, count}] → [{phrase, quantity, source: detected}]."""
    out: list[dict[str, Any]] = []
    for row in detected_items or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("phrase") or "").strip()
        if not name or len(name) < 2:
            continue
        try:
            qty = max(1, int(row.get("count") if row.get("count") is not None else row.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1
        out.append({"phrase": name, "quantity": qty, "source": "detected"})
    return out


def _merge_text_and_detected_items(
    text_items: list[dict[str, Any]],
    detected_items: list[dict[str, Any]] | None,
    *,
    extraction_method: str,
    content_prepared: str = "",
    llm_failed: bool = False,
) -> tuple[list[dict[str, Any]], str]:
    """
    Always merge LLM/text items with vision detectedItems.

    Same stem → text quantity wins (detected row dropped).
    Detected-only stems are appended.
    """
    del content_prepared, llm_failed  # kept for call-site compatibility
    detected = _normalize_detected_items(detected_items)
    if not text_items and not detected:
        return [], extraction_method
    if not text_items:
        return detected, "detected_items"
    if not detected:
        return text_items, extraction_method

    text_stems = {_item_stem(i["phrase"]) for i in text_items}
    merged = list(text_items)
    added_detected = False
    for item in detected:
        stem = _item_stem(item["phrase"])
        if stem in text_stems:
            continue
        merged.append(item)
        text_stems.add(stem)
        added_detected = True
    method = f"{extraction_method}+detected" if added_detected else extraction_method
    return merged, method


def _money(val: Any) -> float | int | None:
    if val is None or val == "":
        return None
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    return int(n) if n == int(n) else round(n, 2)


def _item_type(
    *,
    catalogue_matched: bool,
    source: str | None = None,
    phrase: str = "",
    detected_stems: set[str] | None = None,
) -> str:
    """
    standard = catalogue JSON match OR from vision detectedItems
    custom = neither
    """
    if catalogue_matched:
        return "standard"
    src = (source or "").lower()
    if src == "detected" or "detected" in src:
        return "standard"
    if detected_stems and _item_stem(phrase) in detected_stems:
        return "standard"
    return "custom"


def _detected_stems_from_pricing(pricing: dict[str, Any] | None) -> set[str]:
    stems: set[str] = set()
    for row in (pricing or {}).get("detected_items") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("phrase") or "").strip()
        if name:
            stems.add(_item_stem(name))
    return stems


def build_quote_items(
    pricing: dict[str, Any] | None,
    *,
    collection_side: str | None = None,
    needs_dismantling: bool = False,
) -> list[dict[str, Any]]:
    """
    Flatten pricing line_items into the quote items payload.

    Fields: item_name, quantity, price, inside, inside_with_dismantling, final_price, type.
    type is standard (JSON catalogue or detectedItems) or custom.
    """
    side = (collection_side or "").lower()
    detected_stems = _detected_stems_from_pricing(pricing)
    items_out: list[dict[str, Any]] = []

    for row in (pricing or {}).get("line_items") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("item_name") or row.get("phrase") or "").strip()
        if not name:
            continue
        phrase = str(row.get("phrase") or "")
        # Use catalogue name exactly (e.g. "Paint") — never invent sized labels
        # like "Paint (10 litre)" that don't exist as standard SKUs.
        try:
            qty = max(1, int(row.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1

        price = _money(row.get("base_gbp") if row.get("base_gbp") is not None else row.get("unit_gbp"))
        inside = _money(row.get("inside_surcharge_gbp"))
        dismantle = _money(row.get("inside_dismantling_surcharge_gbp"))

        unit = float(price or 0)
        if side == "inside":
            if needs_dismantling and dismantle is not None:
                unit = float(price or 0) + float(dismantle)
            elif inside is not None:
                unit = float(price or 0) + float(inside)

        line_final = unit * qty
        items_out.append(
            {
                "item_name": name,
                "quantity": qty,
                "price": price,
                "inside": inside,
                "inside_with_dismantling": dismantle,
                "final_price": _money(line_final) if price is not None else None,
                "type": _item_type(
                    catalogue_matched=bool(row.get("item_id") or price is not None),
                    source=str(row.get("source") or ""),
                    phrase=phrase or name,
                    detected_stems=detected_stems,
                ),
            }
        )

    # Unmatched phrases (extracted but no catalogue price)
    for phrase in (pricing or {}).get("unmatched_items") or []:
        label = str(phrase or "").strip()
        if not label:
            continue
        if any(_item_stem(it["item_name"]) == _item_stem(label) for it in items_out):
            continue
        # Find quantity + source from extraction audit if present
        qty = 1
        source = ""
        for entry in (pricing or {}).get("extraction") or []:
            if isinstance(entry, dict) and _item_stem(str(entry.get("phrase") or "")) == _item_stem(label):
                try:
                    qty = max(1, int(entry.get("quantity") or 1))
                except (TypeError, ValueError):
                    qty = 1
                source = str(entry.get("source") or "")
                break
        items_out.append(
            {
                "item_name": label,
                "quantity": qty,
                "price": None,
                "inside": None,
                "inside_with_dismantling": None,
                "final_price": None,
                "type": _item_type(
                    catalogue_matched=False,
                    source=source,
                    phrase=label,
                    detected_stems=detected_stems,
                ),
            }
        )

    return items_out


def lookup_prices_for_text(
    content_main: str,
    *,
    collection_side: str | None = None,
    needs_dismantling: bool = False,
    detected_items: list[dict[str, Any]] | None = None,
    subject: str = "",
) -> dict[str, Any] | None:
    """
    contentMain → items (+ paint size override / LLM / detectedItems) → catalogue prices.

    detected_items (optional): [{name, count}, ...] from vision. Merged with text
    extraction (contentMain wins on conflict); priced from JSON first.
    """
    prepared = prepare_content_main(content_main)
    llm_items, extraction_method, llm_error = extract_items_with_paint_size_override(
        content_main,
        subject=subject,
        detected_items=detected_items,
    )
    llm_failed = bool(llm_error) or not llm_items
    # When LLM returns nothing, always try freeform/regex fallback (quantities optional).
    # message_has_written_item_list is for vision merge policy elsewhere — do not block fallback.
    if not llm_items:
        fallback_items = extract_items_fallback(content_main)
        if fallback_items:
            llm_items = fallback_items
            sources = {str(i.get("source") or "") for i in fallback_items}
            if "nx_fallback" in sources:
                extraction_method = "nx_fallback"
            elif "freeform_fallback" in sources:
                extraction_method = "freeform_fallback"
            else:
                extraction_method = "regex_fallback"
            llm_error = None
            llm_failed = False
        elif message_has_written_item_list(prepared):
            # Customer clearly listed items but we still have nothing — keep llm_error for logs
            pass

    # Paint size list is authoritative — do not merge vision counts (often under-counted)
    if extraction_method != "paint_size_list":
        llm_items, extraction_method = _merge_text_and_detected_items(
            llm_items,
            detected_items,
            extraction_method=extraction_method,
            content_prepared=prepared,
            llm_failed=llm_failed,
        )

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
                    "source": item.get("source"),
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
            "source": item.get("source") or extraction_method,
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
    detected_audit = [
        {"name": d.get("name") or d.get("phrase"), "count": d.get("count") or d.get("quantity")}
        for d in (detected_items or [])
        if isinstance(d, dict)
    ]
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
            "detected_items": detected_audit,
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
        # Same catalogue SKU → one line (e.g. all paint size groups → single "Paint")
        merge_key = str(row.get("item_id") or entry["phrase"])
        if merge_key in merged:
            merged[merge_key]["quantity"] += qty
            # Prefer detected source if either copy came from vision
            prev_src = str(merged[merge_key].get("source") or "")
            new_src = str(entry.get("source") or "")
            if new_src == "detected" or "detected" in new_src:
                merged[merge_key]["source"] = new_src
            elif not prev_src and new_src:
                merged[merge_key]["source"] = new_src
        else:
            merged[merge_key] = {
                **row,
                "quantity": qty,
                # Prefer catalogue name so frontend matches the standard item
                "phrase": row.get("item_name") or entry["phrase"],
                "source": entry.get("source"),
            }

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
                "source": row.get("source"),
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
        "detected_items": detected_audit,
    }
    return apply_collection_pricing(
        result,
        collection_side=collection_side,
        needs_dismantling=needs_dismantling,
    )

