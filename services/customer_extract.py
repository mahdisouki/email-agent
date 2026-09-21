"""Extract structured customer/booking details from a quote conversation — LLM only.

Items are supplied by the frontend (not guessed here).
"""
from __future__ import annotations

import json
import re
from typing import Any

from services.reply_suggester import parse_form_fields
from services.standard_items import enrich_order_items
from services.task_availability import resolve_customer_booking_schedule


def _empty_customer() -> dict[str, Any]:
    return {
        "firstName": None,
        "lastName": None,
        "phoneNumber": None,
        "email": None,
        "postcode": None,
        "address": None,
    }


def _parse_json_object(text: str) -> dict[str, Any] | None:
    raw = (text or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _is_quotation_form_body(text: str) -> bool:
    low = (text or "").lower()
    return "first name:" in low and ("last name:" in low or "comments:" in low)


def _normalize_uk_postcode(raw: str) -> str | None:
    text = (raw or "").strip().upper()
    if not text:
        return None
    compact = re.sub(r"\s+", "", text)
    m = re.search(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)(\d[A-Z]{2})$", compact)
    if not m:
        return text if len(compact) >= 5 else None
    return f"{m.group(1)} {m.group(2)}"


def _postcode_from_text(text: str) -> str | None:
    m = re.search(
        r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b",
        (text or ""),
        re.I,
    )
    return _normalize_uk_postcode(m.group(1)) if m else None


def _name_from_quotation_subject(subject: str) -> tuple[str | None, str | None]:
    m = re.search(
        r"new quotation request from\s+(.+?)\s*$",
        (subject or "").strip(),
        re.I,
    )
    if not m:
        return None, None
    parts = [p for p in m.group(1).strip().split() if p]
    if len(parts) >= 2:
        return parts[0].title(), " ".join(p.title() for p in parts[1:])
    if len(parts) == 1:
        return parts[0].title(), None
    return None, None


def _extract_form_customer_from_thread(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> dict[str, str | None]:
    """Parse quotation-form fields from any thread message (including LWM outbound forms)."""
    out: dict[str, str | None] = {
        "firstName": None,
        "lastName": None,
        "phoneNumber": None,
        "email": None,
        "postcode": None,
        "address": None,
    }
    key_map = {
        "first_name": "firstName",
        "last_name": "lastName",
        "email": "email",
        "phone": "phoneNumber",
        "address": "address",
    }

    rows: list[dict[str, Any]] = list(thread or [])
    if content_main:
        rows.append({"contentMain": content_main, "from": from_header})

    for msg in rows:
        body = (msg.get("contentMain") or msg.get("snippet") or "").strip()
        if not body or not _is_quotation_form_body(body):
            continue
        parsed = parse_form_fields(body)
        for src, dest in key_map.items():
            val = (parsed.get(src) or "").strip()
            if val and not out.get(dest):
                out[dest] = val
        if not out.get("postcode"):
            pc = _postcode_from_text(parsed.get("address") or body)
            if pc:
                out["postcode"] = pc

    return out


def _enrich_customer_from_rules(
    customer: dict[str, Any],
    *,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
    subject: str,
) -> dict[str, Any]:
    """Fill gaps the LLM missed using quotation forms and subject lines."""
    merged = dict(customer or _empty_customer())
    rules = _extract_form_customer_from_thread(thread, content_main, from_header)
    sub_first, sub_last = _name_from_quotation_subject(subject)

    for key in ("firstName", "lastName", "phoneNumber", "email", "postcode", "address"):
        if not (merged.get(key) or "").strip():
            val = rules.get(key)
            if val:
                merged[key] = val

    if not (merged.get("firstName") or "").strip() and sub_first:
        merged["firstName"] = sub_first
    if not (merged.get("lastName") or "").strip() and sub_last:
        merged["lastName"] = sub_last

    if not (merged.get("postcode") or "").strip():
        for source in (
            merged.get("address"),
            content_main,
            subject,
        ):
            pc = _postcode_from_text(str(source or ""))
            if pc:
                merged["postcode"] = pc
                break

    return merged


def _customer_inbound_bodies(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> list[str]:
    try:
        from core.gmail_message import is_lwm_address
    except ImportError:

        def is_lwm_address(_: str) -> bool:
            return False

    bodies: list[str] = []
    seen: set[str] = set()
    rows = list(thread or [])
    if content_main:
        rows.append(
            {"contentMain": content_main, "from": from_header, "direction": "inbound"}
        )
    for msg in rows:
        direction = (msg.get("direction") or "").strip().lower()
        from_h = msg.get("from") or msg.get("from_address") or ""
        if direction == "outbound" or is_lwm_address(from_h):
            continue
        body = (msg.get("contentMain") or msg.get("snippet") or "").strip()
        if not body:
            continue
        key = body[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        bodies.append(body)
    return bodies


def _format_thread_for_llm(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> str:
    rows: list[dict[str, Any]] = list(thread or [])
    if content_main:
        rows.append({"contentMain": content_main, "from": from_header, "direction": "inbound"})
    blocks: list[str] = []
    seen: set[str] = set()
    for i, msg in enumerate(rows):
        body = (msg.get("contentMain") or msg.get("snippet") or "").strip()
        if not body:
            continue
        key = body[:200].lower()
        if key in seen:
            continue
        seen.add(key)
        direction = (msg.get("direction") or "unknown").strip()
        from_h = (msg.get("from") or msg.get("from_address") or "").strip()
        subject = (msg.get("subject") or "").strip()
        header = f"Message {i + 1} [{direction}]"
        if from_h:
            header += f" from {from_h}"
        if subject:
            header += f" — {subject}"
        blocks.append(f"{header}\n{body[:4000]}")
    return "\n\n---\n\n".join(blocks)


def _normalize_customer_payload(raw: dict[str, Any]) -> dict[str, Any]:
    customer_raw = raw.get("customer")
    if not isinstance(customer_raw, dict):
        customer_raw = raw

    def _str(key: str, *aliases: str) -> str | None:
        for name in (key, *aliases):
            val = customer_raw.get(name)
            if val is None and name in raw:
                val = raw.get(name)
            if val is not None and str(val).strip():
                return str(val).strip()
        return None

    booking_date = _str("bookingDate", "booking_date", "collectionDate", "collection_date")
    booking_slot = _str(
        "bookingTimeSlot",
        "booking_time_slot",
        "timeSlot",
        "time_slot",
        "collectionTimeSlot",
        "collection_time_slot",
    )
    if booking_slot and booking_slot.lower().replace(" ", "") == "anytime":
        booking_slot = "AnyTime"

    note = _str(
        "customerNote",
        "customer_note",
        "note",
        "collectionNote",
        "collection_note",
    )

    position = _normalize_position(
        raw.get("position")
        or raw.get("collectionPosition")
        or raw.get("collection_position")
        or raw.get("collectionSide")
        or raw.get("collection_side")
    )

    return {
        "customer": {
            "firstName": _str("firstName", "first_name"),
            "lastName": _str("lastName", "last_name"),
            "phoneNumber": _str("phoneNumber", "phone", "phone_number"),
            "email": _str("email"),
            "postcode": _str("postcode", "post_code"),
            "address": _str("address"),
        },
        "bookingDate": booking_date,
        "bookingTimeSlot": booking_slot,
        "customerNote": note,
        "position": position,
        # Never take items from the LLM — frontend supplies them
        "items": [],
    }


_POSITION_OUTSIDE = "Outside"
_POSITION_INSIDE = "Inside"
_POSITION_INSIDE_DISMANTLING = "Inside with dismantling"


def _normalize_position(value: Any) -> str | None:
    """Map free text / aliases → Outside | Inside | Inside with dismantling."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    low = raw.lower().replace("_", " ").replace("-", " ")
    low = re.sub(r"\s+", " ", low)

    if "dismantl" in low and "no dismantl" not in low:
        return _POSITION_INSIDE_DISMANTLING
    if low in ("outside", "out", "kerbside", "curbside", "exterior"):
        return _POSITION_OUTSIDE
    if low in ("inside", "in", "interior"):
        return _POSITION_INSIDE
    if "outside" in low or "kerbside" in low or "left outside" in low:
        return _POSITION_OUTSIDE
    if "inside" in low:
        return _POSITION_INSIDE
    return None


def _position_from_text(text: str) -> str | None:
    """Infer collection position from customer wording (rules)."""
    low = (text or "").lower()
    needs_dismantling = bool(re.search(r"\bdismantl", low)) and not bool(
        re.search(r"\bno\s+dismantl", low)
    )
    if re.search(
        r"\bno\s+(?:special\s+)?access\s+(?:restrictions?|issues?|problems?)\b|"
        r"\b(?:unrestricted|easy)\s+access\b|"
        r"\bno\s+(?:inside\s+)?access\s+(?:needed|required)\b|"
        r"\b(?:outside only|from outside|outside the property|left outside|kerbside)\b|"
        r"\b(?:on|in) (?:the )?(?:drive(?:way)?|drive way)\b|"
        r"\bin the garden\b|\bfront garden\b|\brear garden\b",
        low,
    ):
        return _POSITION_OUTSIDE
    if re.search(
        r"\b(?:inside the (?:property|house|flat|premises)|from inside|"
        r"in the house|in the flat|in the property|inside collection)\b|"
        r"\b(?:\d+(?:st|nd|rd|th)|first|second|ground)\s+floor\b|"
        r"\b(?:flat|maisonette|apartment)\b",
        low,
    ):
        if needs_dismantling:
            return _POSITION_INSIDE_DISMANTLING
        return _POSITION_INSIDE
    if needs_dismantling:
        return _POSITION_INSIDE_DISMANTLING
    return None


def _merge_position(
    rule_position: str | None,
    llm_position: str | None,
) -> str:
    """Prefer LLM when present; else rules; default Outside."""
    return llm_position or rule_position or _POSITION_OUTSIDE


def _apply_position_to_items(
    items: list[dict[str, Any]],
    position: str,
) -> list[dict[str, Any]]:
    """Attach position to each item; keep per-item client value when already set."""
    out: list[dict[str, Any]] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        existing = _normalize_position(entry.get("position"))
        out.append({**entry, "position": existing or position})
    return out


def normalize_client_order_items(items: list[Any] | None) -> list[dict[str, Any]]:
    """
    Map frontend quote items into order items.
    Does not invent items — only normalizes what the client sent.
    """
    raw: list[dict[str, Any]] = []
    client_status: list[str | None] = []

    for row in items or []:
        if isinstance(row, str) and row.strip():
            raw.append({"phrase": row.strip(), "quantity": 1})
            client_status.append(None)
            continue
        if not isinstance(row, dict):
            continue
        phrase = str(
            row.get("phrase")
            or row.get("item_name")
            or row.get("itemName")
            or row.get("name")
            or row.get("item")
            or ""
        ).strip()
        if not phrase:
            continue
        try:
            qty = max(1, int(row.get("quantity") or row.get("qty") or row.get("count") or 1))
        except (TypeError, ValueError):
            qty = 1
        entry: dict[str, Any] = {"phrase": phrase, "quantity": qty}
        if row.get("item_id") is not None:
            entry["item_id"] = str(row.get("item_id"))
        if row.get("item_name"):
            entry["item_name"] = str(row.get("item_name"))
        for key in ("price", "inside", "inside_with_dismantling", "final_price"):
            if row.get(key) is not None:
                entry[key] = row[key]
        pos = _normalize_position(row.get("position") or row.get("collection_side"))
        if pos:
            entry["position"] = pos
        raw.append(entry)

        status_hint = row.get("status") or row.get("type")
        if isinstance(status_hint, str) and status_hint.strip():
            hint = status_hint.strip().lower()
            if hint in ("standard", "standard_item"):
                client_status.append("standard_item")
            elif hint == "custom":
                client_status.append("custom")
            else:
                client_status.append(hint)
        else:
            client_status.append(None)

    enriched = enrich_order_items(raw)
    out: list[dict[str, Any]] = []
    for i, entry in enumerate(enriched):
        src = raw[i] if i < len(raw) else {}
        if i < len(client_status) and client_status[i]:
            entry = {**entry, "status": client_status[i]}
        if src.get("item_name") and not entry.get("item_name"):
            entry["item_name"] = src["item_name"]
        if src.get("item_id") is not None and not entry.get("item_id"):
            entry["item_id"] = str(src["item_id"])
        for key in ("price", "inside", "inside_with_dismantling", "final_price"):
            if src.get(key) is not None:
                entry[key] = src[key]
        if src.get("position"):
            entry["position"] = src["position"]
        out.append(entry)
    return out


def _merge_booking_schedule(
    rule_schedule: dict[str, str | None],
    llm_schedule: dict[str, str | None],
) -> dict[str, str | None]:
    return {
        "bookingDate": rule_schedule.get("bookingDate")
        or llm_schedule.get("bookingDate"),
        "bookingTimeSlot": rule_schedule.get("bookingTimeSlot")
        or llm_schedule.get("bookingTimeSlot"),
    }


def _fallback_customer_note(customer_bodies: list[str]) -> str | None:
    """Use the first substantive customer message when the LLM is unavailable."""
    for body in customer_bodies:
        text = re.sub(r"\s+", " ", (body or "").strip())
        if len(text) >= 12:
            return text[:600]
    return None


def llm_extract_customer_from_conversation(
    *,
    thread_summary: str,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        from core.llm_service import is_llm_suggest_enabled, llm_chat
    except ImportError:
        return None, "llm_service not available"

    if not is_llm_suggest_enabled():
        return None, "LLM disabled — set OLLAMA_GENERATE_URL and USE_LLM_SUGGEST=true"

    prompt = f"""You extract structured booking details from a UK waste-collection email thread.

Read the ENTIRE conversation below (oldest message first). Merge information from every relevant message.
The customer may send details across several emails (address in one, phone in another).

IMPORTANT: Do NOT extract or invent a list of items to collect. Items are provided separately by the client.
Omit "items" from your JSON (or set "items": []).

Return ONLY JSON:
{{
  "customer": {{
    "firstName": "string or null",
    "lastName": "string or null",
    "phoneNumber": "UK mobile/landline digits only starting with 0, or null",
    "email": "customer email or null",
    "postcode": "UK postcode e.g. SW9 9LN or null",
    "address": "full street collection address or null"
  }},
  "items": [],
  "bookingDate": "YYYY-MM-DD collection date the customer chose, or null",
  "bookingTimeSlot": "AnyTime | 7am-12pm | 12pm-5pm — slot they confirmed, or null",
  "customerNote": "string or null — short note of what they need collected and any special requests",
  "position": "Outside | Inside | Inside with dismantling — where items are collected from, or null"
}}

Rules:
- Extract customer details from inbound/customer messages and quotation forms they submitted.
- Quotation form notifications (often outbound from LWM) include First Name and Last Name fields —
  always merge both names from the form even if later emails only sign off with a first name.
- Parse names from email signatures (e.g. "Victoria Lucas Gallery Assistant") or form fields (First Name / Last Name).
- The email subject "New Quotation Request from First Last" is a reliable source for firstName and lastName.
- Parse addresses even if informal ("Atherfold rd SW99LN", "28 central ave hounslow Tw32qh").
- Always leave items as [] — never invent or list collectible items.
- Do not invent data; use null when unknown.
- UK postcodes: normalize with a space before the last 3 characters when possible (SW99LN → SW9 9LN).
- Phone: strip spaces and formatting; keep leading 0.
- bookingDate / bookingTimeSlot: merge from the whole thread — date may be in an earlier
  message (e.g. "18 June morning") and the slot in a later one (e.g. "7am-12pm would be fine").
- Map morning → 7am-12pm, afternoon/evening → 12pm-5pm, any time → AnyTime.
- position: where the crew collects the items.
  - "Outside" = left outside, kerbside, driveway, garden, no inside access needed.
  - "Inside" = inside the property / house / flat / upstairs, without dismantling.
  - "Inside with dismantling" = inside AND the customer mentions dismantling / taking apart.
  Use null only if the thread gives no access/location clue at all.
- customerNote: summarise what the customer wants collected (usually from their FIRST email)
  and any special instructions from early messages (e.g. "everything will be left outside",
  "please don't make noise", "need it gone urgently"). One or two sentences, British English.
  Do NOT repeat address, phone, weight, floor/access answers, or booking date/time here.
  Use null only if the thread has no item description or special request at all.

CONVERSATION:
{thread_summary[:14000]}
"""

    try:
        raw = llm_chat(
            system=(
                "You read UK waste-collection email threads and extract customer contact, "
                "booking details, and collection position only. Never invent an items list. "
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
    return _normalize_customer_payload(parsed), None


def extract_customer_from_conversation(
    *,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str = "",
    subject: str = "",
    items: list[Any] | None = None,
) -> dict[str, Any]:
    """LLM extraction for customer/booking; items come from the frontend."""
    customer_bodies = _customer_inbound_bodies(thread, content_main, from_header)
    rule_schedule = resolve_customer_booking_schedule(customer_bodies)
    thread_text = "\n\n".join(customer_bodies) if customer_bodies else (content_main or "")
    rule_position = _position_from_text(thread_text)
    thread_summary = _format_thread_for_llm(thread, content_main, from_header)
    llm_result, llm_error = llm_extract_customer_from_conversation(
        thread_summary=thread_summary,
    )
    provided_items = normalize_client_order_items(items)

    if llm_result:
        llm_schedule = {
            "bookingDate": llm_result.get("bookingDate"),
            "bookingTimeSlot": llm_result.get("bookingTimeSlot"),
        }
        booking = _merge_booking_schedule(rule_schedule, llm_schedule)
        customer = _enrich_customer_from_rules(
            llm_result["customer"],
            thread=thread,
            content_main=content_main,
            from_header=from_header,
            subject=subject,
        )
        position = _merge_position(rule_position, llm_result.get("position"))
        return {
            "customer": customer,
            "items": _apply_position_to_items(provided_items, position),
            "position": position,
            "bookingDate": booking.get("bookingDate"),
            "bookingTimeSlot": booking.get("bookingTimeSlot"),
            "customerNote": llm_result.get("customerNote"),
            "source": "llm",
            "llm_error": None,
            "subject": subject or None,
        }

    position = _merge_position(rule_position, None)
    return {
        "customer": _empty_customer(),
        "items": _apply_position_to_items(provided_items, position),
        "position": position,
        "bookingDate": rule_schedule.get("bookingDate"),
        "bookingTimeSlot": rule_schedule.get("bookingTimeSlot"),
        "customerNote": _fallback_customer_note(customer_bodies),
        "source": "llm",
        "llm_error": llm_error,
        "subject": subject or None,
    }
