"""Prepare contentMain text for LLM item extraction."""
from __future__ import annotations

import re

def _comments_from_form(text: str) -> str:
    m = re.search(
        r"Comments:\s*(.+?)(?=\s+Uploaded Items\b|\s+Thank you for using|$)",
        text,
        re.I | re.S,
    )
    return m.group(1).strip() if m else ""


def prepare_content_main(content_main: str) -> str:
    """Use customer wording from contentMain; prefer Comments on quotation forms.

    Prefer text above any quoted LWM email so item lists stay clean for the LLM.
    """
    text = (content_main or "").strip()
    if not text:
        return ""
    low = text.lower()
    if "first name:" in low and "comments:" in low:
        text = _comments_from_form(text) or text
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


