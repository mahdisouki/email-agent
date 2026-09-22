"""Prepare contentMain text for LLM item extraction."""
from __future__ import annotations

import re


def _body_from_form_field(text: str, field: str) -> str:
    """Pull Comments: / Message: body from LWM form notifications."""
    m = re.search(
        rf"{re.escape(field)}:\s*(.+?)(?="
        r"\s+Uploaded Items\b|"
        r"\s+Submitted:|"
        r"\s+Thank you for using\b|"
        r"\s+London Waste Management\s*$|"
        r"$)",
        text,
        re.I | re.S,
    )
    return m.group(1).strip() if m else ""


def _customer_body_from_form(text: str) -> str:
    """
    Quotation forms use Comments:; contact forms use Message:.
    Prefer Comments, then Message.
    """
    return _body_from_form_field(text, "Comments") or _body_from_form_field(text, "Message")


def prepare_content_main(content_main: str) -> str:
    """Use customer wording from contentMain; prefer Comments/Message on LWM forms.

    Prefer text above any quoted LWM email so item lists stay clean for the LLM.
    """
    text = (content_main or "").strip()
    if not text:
        return ""
    low = text.lower()
    if "first name:" in low and ("comments:" in low or "message:" in low):
        text = _customer_body_from_form(text) or text
    # Drop quoted outbound LWM / reply history after the customer's own message
    text = re.split(
        r"\bFrom:\s*[\"']?London Waste Management\b|"
        r"\bFrom:\s*[\"']?hello@londonwastemanagement\b|"
        r"\bSent:\s*.{0,80}?\bFrom:\s*[\"']?London Waste Management\b|"
        r"\bon .+?wrote:|"
        r"\blondon waste management support\b.*\bwrote:|"
        r"\b-{2,}\s*Original Message\s*-{2,}",
        text,
        maxsplit=1,
        flags=re.I | re.S,
    )[0].strip()
    text = re.sub(r"\b\d+\s+photos?\s+have\s+been\s+sent\.?\s*$", "", text, flags=re.I).strip()
    return text.strip()
