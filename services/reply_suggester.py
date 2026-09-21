"""Route suggest_reply by category (quote, complaint, close, default)."""
import re
from typing import Any

from core.gmail_message import display_name_first, is_lwm_address, parse_email_address
from core.llm_service import cap_suggestions
from services.classifier import classify_email

CLOSE_MARKERS = (
    "council will collect",
    "council are collecting",
    "no longer need",
    "don't need your service",
    "do not need your service",
    "cancel my request",
    "please cancel",
    "apologies for wasting your time",
)

FORM_FIELD_LABELS = (
    "First Name",
    "Last Name",
    "Address",
    "Email",
    "Phone Number",
    "Comments",
    "Uploaded Items",
)


def field(text: str, label: str) -> str | None:
    others = "|".join(
        re.escape(f) for f in FORM_FIELD_LABELS if f.lower() != label.lower()
    )
    inline = re.search(
        rf"(?:^|\s){re.escape(label)}:\s*(.+?)"
        rf"(?=\s+(?:{others})(?:\s*\([^)]*\))?\s*:|\s+Thank you for using|$)",
        text,
        re.I | re.S,
    )
    if inline:
        return inline.group(1).strip()
    line = re.search(rf"^{re.escape(label)}:\s*(.+)$", text, re.I | re.M)
    return line.group(1).strip() if line else None


def parse_form_fields(text: str) -> dict[str, str]:
    return {
        k: v
        for k, v in (
            ("first_name", field(text, "First Name")),
            ("last_name", field(text, "Last Name")),
            ("email", field(text, "Email")),
            ("phone", field(text, "Phone Number")),
            ("address", field(text, "Address")),
            ("comments", field(text, "Comments")),
        )
        if v
    }


def thread_text(thread: list[dict[str, Any]] | None) -> str:
    if not thread:
        return ""
    parts = []
    for msg in thread:
        parts.append(msg.get("subject") or "")
        parts.append(msg.get("contentMain") or "")
    return "\n".join(parts)


def _strip_quoted_reply(text: str) -> str:
    parts = re.split(
        r"\bFrom:\s*London Waste Management\b|\bFrom:\s*hello@londonwastemanagement\b|"
        r"\bon .+?wrote:|\blondon waste management support\b.*\bwrote:",
        text,
        maxsplit=1,
        flags=re.I | re.S,
    )
    return parts[0].strip()


def _name_from_email_address(from_address: str) -> str | None:
    if not from_address or "@" not in from_address:
        return None
    local = from_address.split("@", 1)[0]
    local = re.sub(r"[._-]+", " ", local).strip()
    if not local:
        return None
    first = local.split()[0]
    if len(first) >= 2 and first.isalpha():
        return first.title()
    return None


_NAME_BLOCKLIST = frozenset(
    {
        "park", "road", "clinical", "manager", "note", "thank", "dear", "south", "west",
        "london", "springfield", "trinity", "ground", "floor", "subject", "waste",
        "management", "support", "head", "operations", "strategy", "ndt", "camhs",
    }
)


def _name_from_signature(text: str) -> str | None:
    for m in re.finditer(r"\b([A-Z][a-z]{2,})\s+[A-Z][a-z][A-Za-z' -]{2,40}", text):
        first = m.group(1).lower()
        if first not in _NAME_BLOCKLIST:
            return m.group(1).title()
    return None


def _format_greeting_token(token: str) -> str:
    if token.isupper() and len(token) <= 4:
        return token
    return token.title()


def _greeting_name_from_form(form: dict[str, str]) -> str | None:
    first = (form.get("first_name") or "").strip()
    last = (form.get("last_name") or "").strip()
    if last:
        token = last.split()[0]
        if token.lower() not in _NAME_BLOCKLIST:
            return _format_greeting_token(token)
    if first:
        token = first.split()[0]
        if token.lower() not in _NAME_BLOCKLIST:
            return _format_greeting_token(token)
    return None


def _name_from_quotation_subject(subject: str) -> str | None:
    m = re.search(r"new quotation request from\s+(.+?)\s*$", subject or "", re.I)
    if not m:
        return None
    parts = m.group(1).strip().split()
    if not parts:
        return None
    first_tok, last_tok = parts[0], parts[-1]
    if last_tok.lower() not in _NAME_BLOCKLIST:
        return _format_greeting_token(last_tok)
    if first_tok.lower() not in _NAME_BLOCKLIST and last_tok != first_tok:
        return _format_greeting_token(first_tok)
    return None


def _merge_form_fields(
    parsed_form: dict | None,
    content_main: str,
    subject: str,
    thread: list[dict] | None,
) -> dict[str, str]:
    merged: dict[str, str] = dict(parsed_form or {})
    for source in (content_main, subject, thread_text(thread)):
        if not source:
            continue
        for key, value in parse_form_fields(source).items():
            merged.setdefault(key, value)
    return merged


def first_name(
    parsed_form: dict | None,
    content_main: str,
    thread: list[dict] | None,
    subject: str = "",
    from_header: str = "",
) -> str:
    form = _merge_form_fields(parsed_form, content_main, subject, thread)
    greeting = _greeting_name_from_form(form)
    if greeting:
        return greeting

    subject_name = _name_from_quotation_subject(subject)
    if subject_name:
        return subject_name

    customer_text = _strip_quoted_reply(content_main)
    sig_name = _name_from_signature(customer_text)
    if sig_name:
        return sig_name

    if not is_lwm_address(from_header):
        from_display = display_name_first(from_header)
        if from_display and from_display.lower() not in _NAME_BLOCKLIST:
            return from_display

        from_email = _name_from_email_address(parse_email_address(from_header))
        if from_email and from_email.lower() not in _NAME_BLOCKLIST:
            return from_email

    m = re.search(r"dear\s+([A-Za-z]+)", content_main, re.I)
    if m and m.group(1).lower() not in _NAME_BLOCKLIST:
        return m.group(1).title()

    return "there"


def is_close_message(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in CLOSE_MARKERS)


def return_replies(
    result: dict,
    replies: list[str],
    *,
    draft_source: str = "rules",
) -> dict:
    result.pop("suggested_reply", None)
    result.pop("_pricing", None)
    result["suggested_replies"] = cap_suggestions(replies)
    result["draft_source"] = draft_source
    return result


def _default_reply(category: str, name: str) -> dict:
    """No drafts for unsupported categories (e.g. other)."""
    return return_replies(
        {
            "category": category,
            "phase": "unsupported",
            "missing_slots": [],
            "reason": f"No dedicated reply flow for '{category}'",
            "customer_name": name,
        },
        [],
        draft_source="none",
    )


def _pending_rebuild(category: str, name: str) -> dict:
    """Quote / complaint suggest flows removed — empty drafts until reimplemented."""
    return return_replies(
        {
            "category": category,
            "phase": "pending_rebuild",
            "missing_slots": [],
            "reason": f"{category} suggest reply not implemented yet",
            "customer_name": name,
        },
        [],
        draft_source="none",
    )


def suggest_reply(
    subject: str = "",
    content_main: str = "",
    snippet: str = "",
    from_header: str = "",
    direction: str = "inbound",
    thread: list[dict[str, Any]] | None = None,
    parsed_form: dict | None = None,
    attachments: list[dict] | None = None,
    category: str | None = None,
    detected_items: list[dict[str, Any]] | None = None,
) -> dict:
    subject = (subject or "").strip()
    content_main = (content_main or "").strip()
    snippet = (snippet or "").strip()
    thread = thread or []

    if not content_main:
        return return_replies(
            {
                "category": category or "other",
                "phase": "error",
                "missing_slots": [],
                "reason": "contentMain is required",
            },
            [],
        )

    from_email = parse_email_address(from_header)
    if not category:
        classified = classify_email(
            subject, content_main, snippet=snippet, from_address=from_email
        )
        category = classified["category"]

    combined = f"{subject}\n{snippet}\n{content_main}\n{thread_text(thread)}"
    parsed_form = _merge_form_fields(parsed_form, content_main, subject, thread)
    name = first_name(parsed_form, content_main, thread, subject=subject, from_header=from_header)

    if is_close_message(combined):
        reply = (
            f"Hi {name},\n\n"
            "Thank you for letting us know.\n\n"
            "No problem at all — please get in touch if you need anything in the future.\n\n"
            "Kind regards."
        )
        return return_replies(
            {
                "category": category,
                "phase": "close",
                "missing_slots": [],
                "reason": "Customer indicated they no longer need the service",
            },
            [reply],
        )

    if category == "quote":
        from flows.quote import suggest_quote_reply

        return suggest_quote_reply(
            content_main=content_main,
            subject=subject,
            from_header=from_header,
            parsed_form=parsed_form,
            attachments=attachments,
            thread=thread,
            greeting_name=name,
            detected_items=detected_items,
        )

    if category == "complaint":
        from flows.complaint import suggest_complaint_reply

        return suggest_complaint_reply(
            content_main=content_main,
            subject=subject,
            from_header=from_header,
            thread=thread,
            greeting_name=name,
        )

    return _default_reply(category, name)
