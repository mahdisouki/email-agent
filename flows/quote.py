"""
Quote suggest reply — extract items + catalogue prices only (no email drafts).

Returns suggested_replies=[] and an items[] list with outside / inside /
inside_with_dismantling fees and a final_price from the thread access context.
"""
from __future__ import annotations

import re
from typing import Any

from core.gmail_message import is_lwm_address
from services.reply_suggester import field, parse_form_fields, return_replies
from services.standard_items import build_quote_items, lookup_prices_for_text


def _strip_customer_reply(text: str) -> str:
    parts = re.split(
        r"\bFrom:\s*London Waste Management\b|\bFrom:\s*hello@londonwastemanagement\b|"
        r"\bon .+?wrote:|\blondon waste management support\b.*\bwrote:",
        text or "",
        maxsplit=1,
        flags=re.I | re.S,
    )
    return parts[0].strip()


def _is_form_body(text: str) -> bool:
    low = (text or "").lower()
    return "first name:" in low and (
        "comments:" in low or "phone number:" in low or "email:" in low
    )


def _customer_reply_bodies(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> list[str]:
    bodies: list[str] = []
    seen: set[str] = set()
    rows = list(thread or [])
    if content_main and not _is_form_body(content_main):
        rows.append({"contentMain": content_main, "from": from_header})
    for msg in rows:
        body = _strip_customer_reply((msg.get("contentMain") or "").strip())
        from_h = msg.get("from") or msg.get("from_address") or ""
        if not body or _is_form_body(body) or is_lwm_address(from_h):
            continue
        key = body[:240].lower()
        if key in seen:
            continue
        seen.add(key)
        bodies.append(body)
    return bodies


def _form_body_from_context(
    thread: list[dict[str, Any]] | None,
    content_main: str,
) -> str:
    for msg in thread or []:
        body = (msg.get("contentMain") or "").strip()
        if _is_form_body(body):
            return body
    if _is_form_body(content_main):
        return content_main
    return ""


def _comments_text(text: str) -> str:
    raw = (field(text, "Comments") or "").strip()
    raw = re.sub(r"uploaded items\s*\(\s*\d+\s*\).*", "", raw, flags=re.I | re.S).strip()
    raw = re.sub(r"view image.*", "", raw, flags=re.I | re.S).strip()
    return raw.strip()


def _merge_form(
    parsed_form: dict[str, str] | None,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str = "",
) -> dict[str, str]:
    form = dict(parsed_form or {})
    form_body = _form_body_from_context(thread, content_main)
    if form_body:
        form = {**parse_form_fields(form_body), **form}
    if not form.get("comments"):
        comments = _comments_text(form_body or content_main)
        if comments:
            form["comments"] = comments
    return form


def _pricing_source_text(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    form: dict[str, str],
    from_header: str = "",
) -> str:
    """Combine form comments + all customer messages for item extraction."""
    parts: list[str] = []
    comments = (form.get("comments") or "").strip()
    if comments:
        parts.append(comments)
    for body in _customer_reply_bodies(thread, content_main, from_header):
        if body and body not in parts:
            parts.append(body)
    if content_main and not _is_form_body(content_main):
        stripped = _strip_customer_reply(content_main)
        if stripped and stripped not in parts:
            parts.append(stripped)
    form_body = _form_body_from_context(thread, content_main)
    if form_body and not parts:
        return form_body
    return "\n\n".join(parts) if parts else content_main


def _collection_side_from_text(text: str) -> str | None:
    """outside | inside | None — from customer wording."""
    low = (text or "").lower()
    if re.search(
        r"\bno\s+(?:special\s+)?access\s+(?:restrictions?|issues?|problems?)\b|"
        r"\b(?:unrestricted|easy)\s+access\b|"
        r"\bno\s+(?:inside\s+)?access\s+(?:needed|required)\b|"
        r"\b(?:outside only|from outside|outside the property|left outside|kerbside)\b|"
        r"\b(?:on|in) (?:the )?(?:drive(?:way)?|drive way)\b|"
        r"\bin the garden\b|\bfront garden\b|\brear garden\b",
        low,
    ):
        return "outside"
    if re.search(
        r"\b(?:inside the (?:property|house|flat|premises)|from inside|"
        r"in the house|in the flat|in the property|inside collection)\b|"
        r"\b(?:\d+(?:st|nd|rd|th)|first|second|ground)\s+floor\b|"
        r"\b(?:flat|maisonette|apartment)\b",
        low,
    ):
        return "inside"
    return None


def _needs_dismantling_from_text(text: str) -> bool:
    low = (text or "").lower()
    if re.search(r"\bno\s+dismantl", low):
        return False
    return bool(re.search(r"\bdismantl", low))


def suggest_quote_reply(
    *,
    content_main: str,
    subject: str = "",
    from_header: str = "",
    parsed_form: dict[str, str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    thread: list[dict[str, Any]] | None = None,
    greeting_name: str = "there",
    detected_items: list[dict[str, Any]] | None = None,
) -> dict:
    """Extract + price items from the quote thread. No suggested email drafts."""
    del attachments, greeting_name  # unused in items-only mode

    form = _merge_form(parsed_form, thread, content_main, from_header)
    pricing_text = _pricing_source_text(thread, content_main, form, from_header)
    collection_side = _collection_side_from_text(pricing_text)
    needs_dismantling = _needs_dismantling_from_text(pricing_text)

    pricing = lookup_prices_for_text(
        pricing_text,
        collection_side=collection_side,
        needs_dismantling=needs_dismantling,
        detected_items=detected_items,
        subject=subject,
    )
    items = build_quote_items(
        pricing,
        collection_side=collection_side,
        needs_dismantling=needs_dismantling,
    )

    result = {
        "category": "quote",
        "phase": "items_only",
        "missing_slots": [],
        "reason": "Quote items extracted from email/thread and detectedItems",
        "collection_side": collection_side,
        "needs_dismantling": needs_dismantling,
        "items": items,
        "extraction_method": (pricing or {}).get("extraction_method"),
        "unmatched_items": (pricing or {}).get("unmatched_items") or [],
        "llm_error": (pricing or {}).get("llm_error"),
    }
    return return_replies(result, [], draft_source="items_only")
