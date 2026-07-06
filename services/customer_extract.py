"""Extract structured customer details and items from a quote conversation — LLM only."""
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

    items_out: list[dict[str, Any]] = []
    items_raw = raw.get("items")
    if isinstance(items_raw, list):
        for row in items_raw:
            if isinstance(row, str) and row.strip():
                items_out.append({"phrase": row.strip(), "quantity": 1})
                continue
            if not isinstance(row, dict):
                continue
            phrase = str(
                row.get("phrase") or row.get("item") or row.get("name") or row.get("description") or ""
            ).strip()
            if not phrase:
                continue
            try:
                qty = max(1, int(row.get("quantity") or row.get("qty") or 1))
            except (TypeError, ValueError):
                qty = 1
            items_out.append({"phrase": phrase, "quantity": qty})

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

    return {
        "customer": {
            "firstName": _str("firstName", "first_name"),
            "lastName": _str("lastName", "last_name"),
            "phoneNumber": _str("phoneNumber", "phone", "phone_number"),
            "email": _str("email"),
            "postcode": _str("postcode", "post_code"),
            "address": _str("address"),
        },
        "items": items_out,
        "bookingDate": booking_date,
        "bookingTimeSlot": booking_slot,
        "customerNote": note,
    }


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
The customer may send details across several emails (address in one, phone in another, items in a third).

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
  "items": [
    {{"phrase": "dining chair", "quantity": 2}},
    {{"phrase": "toilet cistern", "quantity": 1}}
  ],
  "bookingDate": "YYYY-MM-DD collection date the customer chose, or null",
  "bookingTimeSlot": "AnyTime | 7am-12pm | 12pm-5pm — slot they confirmed, or null",
  "customerNote": "string or null — short note of what they need collected and any special requests"
}}

Rules:
- Extract customer details from inbound/customer messages and quotation forms they submitted.
- Quotation form notifications (often outbound from LWM) include First Name and Last Name fields —
  always merge both names from the form even if later emails only sign off with a first name.
- Parse names from email signatures (e.g. "Victoria Lucas Gallery Assistant") or form fields (First Name / Last Name).
- The email subject "New Quotation Request from First Last" is a reliable source for firstName and lastName.
- Parse addresses even if informal ("Atherfold rd SW99LN", "28 central ave hounslow Tw32qh").
- items = physical waste, furniture, or materials to collect mentioned anywhere in the thread.
- If photos are mentioned but items are not listed, leave items as [].
- Do not invent data; use null when unknown.
- UK postcodes: normalize with a space before the last 3 characters when possible (SW99LN → SW9 9LN).
- Phone: strip spaces and formatting; keep leading 0.
- bookingDate / bookingTimeSlot: merge from the whole thread — date may be in an earlier
  message (e.g. "18 June morning") and the slot in a later one (e.g. "7am-12pm would be fine").
- Map morning → 7am-12pm, afternoon/evening → 12pm-5pm, any time → AnyTime.
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
                "You read UK waste-collection email threads and extract customer contact "
                "details and items to collect. Output only valid JSON. British English."
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
) -> dict[str, Any]:
    """Pure LLM extraction — no rule-based fallback."""
    customer_bodies = _customer_inbound_bodies(thread, content_main, from_header)
    rule_schedule = resolve_customer_booking_schedule(customer_bodies)
    thread_summary = _format_thread_for_llm(thread, content_main, from_header)
    llm_result, llm_error = llm_extract_customer_from_conversation(
        thread_summary=thread_summary,
    )

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
        return {
            "customer": customer,
            "items": enrich_order_items(llm_result["items"]),
            "bookingDate": booking.get("bookingDate"),
            "bookingTimeSlot": booking.get("bookingTimeSlot"),
            "customerNote": llm_result.get("customerNote"),
            "source": "llm",
            "llm_error": None,
            "subject": subject or None,
        }

    return {
        "customer": _empty_customer(),
        "items": [],
        "bookingDate": rule_schedule.get("bookingDate"),
        "bookingTimeSlot": rule_schedule.get("bookingTimeSlot"),
        "customerNote": _fallback_customer_note(customer_bodies),
        "source": "llm",
        "llm_error": llm_error,
        "subject": subject or None,
    }
