"""Map backend Gmail message payloads to suggest_reply (no HTTP calls)."""
import html as html_module
import re
from typing import Any

LWM_FROM_HINTS = ("londonwastemanagement.com", "hello@londonwastemanagement")
LWM_INBOX_HINTS = ("hello@londonwastemanagement.com",)


def parse_email_address(raw: str) -> str:
    if not raw:
        return ""
    m = re.search(r"<([^>]+)>", raw)
    return (m.group(1) if m else raw).strip()


def parse_display_name(raw: str) -> str:
    """Display name before angle-bracket email, e.g. Jack Phillipson <jack@…> → Jack Phillipson."""
    if not raw:
        return ""
    text = raw.strip()
    if "<" in text:
        text = text.split("<", 1)[0].strip()
    text = text.strip('"').strip("'").strip()
    if not text or "@" in text:
        return ""
    return text


def display_name_first(raw: str) -> str | None:
    """First token of the display name in From."""
    name = parse_display_name(raw)
    if not name:
        return None
    first = name.split()[0]
    if len(first) >= 2 and first.replace("-", "").replace("'", "").isalpha():
        return first.title()
    return None


def is_lwm_address(from_field: str) -> bool:
    low = (from_field or "").lower()
    return any(h in low for h in LWM_FROM_HINTS)


def _message_text(msg: dict[str, Any]) -> str:
    parts = [
        msg.get("contentMain") or "",
        msg.get("snippet") or "",
        msg.get("body") or "",
    ]
    text = "\n".join(p for p in parts if p).strip()
    text = html_module.unescape(text)
    return text


def is_quotation_form_notification(msg: dict[str, Any]) -> bool:
    """
    Website form emails: From/To may show LWM, but body is the customer's New Quotation Request.
    """
    subject = (msg.get("subject") or "").lower()
    if "new quotation request" not in subject and "quotation request" not in subject:
        return False
    text = _message_text(msg).lower()
    return "first name:" in text and (
        "comments:" in text or "phone number:" in text or "email:" in text
    )


def is_lwm_reply_to_customer(msg: dict[str, Any]) -> bool:
    """LWM replying to a customer (not an inbox form notification)."""
    if not is_lwm_address(msg.get("from") or ""):
        return False
    if is_quotation_form_notification(msg):
        return False
    to_raw = (msg.get("to") or "").lower()
    if any(inbox in to_raw for inbox in LWM_INBOX_HINTS):
        return False
    to_addr = parse_email_address(msg.get("to") or "")
    if to_addr and not is_lwm_address(to_addr):
        return True
    return False


def effective_direction(msg: dict[str, Any]) -> str:
    """
    inbound = draft a reply TO the customer (form request, or their email).
    outbound = LWM already wrote to the customer; may still draft with warning.
    """
    if is_quotation_form_notification(msg):
        return "inbound"
    if is_lwm_address(msg.get("from") or ""):
        to_raw = (msg.get("to") or "").lower()
        if any(inbox in to_raw for inbox in LWM_INBOX_HINTS):
            return "inbound"
        if is_lwm_reply_to_customer(msg):
            return "outbound"
        # Phone/email to customer (no To on thread rows) — still staff outbound
        return "outbound"
    return "inbound"


def normalize_message(msg: dict[str, Any]) -> dict[str, Any]:
    from_raw = msg.get("from") or msg.get("from_address") or ""
    body = _message_text(msg)
    direction = effective_direction({**msg, "contentMain": body})
    msg_id = msg.get("id") or msg.get("messageId") or msg.get("message_id")
    msg_id = str(msg_id).strip() if msg_id else None
    return {
        "id": msg_id,
        "messageId": msg_id,
        "threadId": msg.get("threadId") or msg.get("thread_id"),
        "from": from_raw,
        "to": msg.get("to"),
        "subject": (msg.get("subject") or "").strip(),
        "contentMain": body,
        "snippet": msg.get("snippet"),
        "direction": direction,
        "is_quotation_form": is_quotation_form_notification({**msg, "contentMain": body}),
        "aiCategory": msg.get("aiCategory") or msg.get("category"),
        "date": msg.get("date") or msg.get("internalDate"),
    }


def pick_target_message(
    latest: dict[str, Any],
    thread: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], str | None]:
    """Pick the message to draft a customer-facing reply from."""
    candidates: list[dict[str, Any]] = []
    if thread:
        candidates.extend(thread)
    candidates.append(latest)

    # Follow-up: prefer the latest real customer reply over the original form
    customer_msgs: list[dict[str, Any]] = []
    for m in candidates:
        body = _message_text(m)
        from_raw = m.get("from") or m.get("from_address") or ""
        if not body:
            continue
        enriched = {**m, "contentMain": body}
        if is_quotation_form_notification(enriched):
            continue
        if is_lwm_address(from_raw):
            continue
        customer_msgs.append(enriched)

    if customer_msgs:
        return customer_msgs[-1], None

    # First contact: quotation form notification (customer data in body)
    forms = [m for m in candidates if m.get("is_quotation_form") or is_quotation_form_notification(m)]
    if forms:
        return forms[-1], None

    inbound = [m for m in candidates if m.get("direction") == "inbound"]
    if inbound:
        return inbound[-1], None

    if latest.get("direction") == "inbound":
        return latest, None

    # LWM already emailed the customer — still draft (e.g. follow-up template)
    if is_lwm_reply_to_customer(latest):
        return latest, (
            "Latest message is LWM's email to the customer. Draft is a suggested next reply; "
            "confirm thread context if the customer has not answered yet."
        )

    return latest, None


def messages_to_thread(messages: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not messages:
        return None
    return [normalize_message(m) for m in messages]
