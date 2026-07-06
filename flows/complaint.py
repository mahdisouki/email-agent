"""
Complaint suggest reply — LWM admin tone without analysing photos or matching allegations.

Uses fixed rule templates (four distinct options per phase). Staff pick the best fit;
templates acknowledge photos when mentioned but never describe image content.
"""
from __future__ import annotations

import re
from typing import Any

from core.llm_service import cap_suggestions
from services.reply_suggester import return_replies, thread_text


def _extract_order_ref(text: str) -> str | None:
    m = re.search(r"(#?ORD\d+)", text, re.I)
    if m:
        ref = m.group(1).upper()
        return ref if ref.startswith("#") else f"#{ref}"
    return None


def _customer_message_only(text: str) -> str:
    parts = re.split(
        r"\bFrom:\s*London Waste Management\b|\bFrom:\s*hello@londonwastemanagement\b|"
        r"\bon .+?wrote:|\blondon waste management support\b.*\bwrote:",
        text,
        maxsplit=1,
        flags=re.I | re.S,
    )
    return parts[0].strip()


def _lwm_replied_in_thread(thread: list[dict[str, Any]] | None) -> bool:
    if not thread:
        return False
    for msg in thread:
        if (msg.get("direction") or "").lower() != "outbound":
            continue
        body = (msg.get("contentMain") or "").lower()
        if any(
            phrase in body
            for phrase in (
                "internal review",
                "internal investigation",
                "sincerely apologise",
                "taking this seriously",
                "thank you for bringing this",
                "thank you for getting back to me",
            )
        ):
            return True
    return False


def _lwm_quoted_in_body(content_main: str) -> bool:
    low = content_main.lower()
    if "from: london waste management" in low or (
        "hello@londonwastemanagement" in low and "sent:" in low
    ):
        return True
    if "dear " in low and "sincerely apologise" in low and "kind regards" in low:
        return True
    return False


def _is_follow_up(
    *,
    subject: str,
    content_main: str,
    thread: list[dict[str, Any]] | None,
) -> bool:
    if re.search(r"^re:\s", (subject or "").strip(), re.I):
        return True
    if _lwm_quoted_in_body(content_main):
        return True
    if _lwm_replied_in_thread(thread):
        return True
    return False


def _is_service_failure_complaint(text: str) -> bool:
    """Collection/job issues (re-collect) vs driver conduct / general complaints."""
    low = text.lower()
    if re.search(r"#?ord\d+", low):
        return True
    markers = (
        "left on my driveway",
        "left on the driveway",
        "left on driveway",
        "incomplete pickup",
        "incomplete collection",
        "partial collection",
        "forgot part of",
        "remaining items",
        "rubbish left",
        "fly tipping",
        "not collected",
        "reimbursed",
        "refund",
        "sawdust left",
        "mess left",
    )
    return any(m in low for m in markers)


def _variant_investigation_apology(name: str) -> str:
    return (
        f"Dear {name},\n\n"
        "Thank you for taking the time to bring this to our attention. "
        "I am very sorry to hear about your experience, and I completely understand "
        "how unacceptable and frustrating this situation must have been.\n\n"
        "I want to sincerely apologise for the behaviour you have described. "
        "This is not the standard we expect from any member of our team.\n\n"
        "Please rest assured that:\n"
        "1. We are taking this incident extremely seriously. "
        "Your report has been forwarded to management for a full internal review.\n"
        "2. The team involved will be formally questioned and disciplined appropriately.\n"
        "3. We appreciate any reference details you have provided — this helps us identify the crew involved.\n\n"
        "Once our internal investigation is complete, we will update you with the actions taken.\n\n"
        "Kind regards."
    )


def _variant_thank_details(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        "Thank you for providing the details you have shared — these will help us identify "
        "the crew involved and support our internal review.\n\n"
        "We will update you once our investigation is complete.\n\n"
        "Kind regards."
    )


def _variant_recollect(name: str, order_ref: str | None) -> str:
    order_line = f"Regarding {order_ref}, " if order_ref else ""
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        f"{order_line}"
        "our dispatching team has confirmed we will send a team on [DAY] "
        "for the anytime slot (between [TIME_START] and [TIME_END]) to collect any remaining items.\n\n"
        "As long as they are outside and in an accessible location, the team will be able to "
        "collect them with no issues and you do not need to be present.\n\n"
        "Please let me know if you need any further information.\n\n"
        "Kind regards."
    )


def _variant_acknowledge_ask_details(name: str, order_ref: str | None) -> str:
    if order_ref:
        detail = (
            f"I can see you mentioned {order_ref}. "
            "Please confirm this is the correct order reference and let me know "
            "the best contact number in case our team needs to reach you."
        )
    else:
        detail = (
            "To help us resolve this quickly, please confirm the address and date of the collection, "
            "and the best contact number in case our team needs to reach you."
        )
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        "I am sorry to hear you have had a poor experience with our service. "
        f"{detail}\n\n"
        "We will review this as a priority and come back to you with a resolution.\n\n"
        "Kind regards."
    )


def _variant_apology_update(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        "I am sorry to hear about the issues you have experienced. "
        "We are looking into this matter internally and will update you as soon as we have more information.\n\n"
        "Thank you again for bringing this to our attention.\n\n"
        "Kind regards."
    )


def _variant_evidence_follow_up(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for sending over the photo.\n\n"
        "We completely understand your concerns. We are continuing our internal investigation "
        "into the matter you raised, as this is not the standard we expect from our teams. "
        "The photo has been received and passed to management as part of our review.\n\n"
        "Appropriate action will be taken once the review is complete. "
        "You are free to share any information with the relevant authorities if you wish, "
        "and we will cooperate with any formal request.\n\n"
        "Thank you again for bringing this to our attention.\n\n"
        "Kind regards."
    )


def _variant_investigation_update(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        "Our internal investigation is still underway. We take your complaint seriously "
        "and appropriate action will be taken once the review is complete.\n\n"
        "We will update you as soon as we have an outcome.\n\n"
        "Kind regards."
    )


def _variant_recollect_confirm(name: str, order_ref: str | None) -> str:
    order_line = f"For {order_ref}, " if order_ref else ""
    return (
        f"Hi {name},\n\n"
        f"{order_line}We have arranged for a team to return on [DAY] between [TIME_START] and [TIME_END] "
        "to collect any remaining items, provided they are outside and accessible.\n\n"
        "Please let me know if anything changes before then.\n\n"
        "Kind regards."
    )


def _variant_resolution_offer(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for your patience.\n\n"
        "We are sorry for the inconvenience caused. Our team is working to put this right "
        "and will confirm the next steps with you shortly, including any re-collection or "
        "refund review if applicable.\n\n"
        "Please let me know if you need anything else in the meantime.\n\n"
        "Kind regards."
    )


def _four_variants_first_contact(
    name: str, order_ref: str | None, *, service_failure: bool
) -> list[str]:
    if service_failure:
        return [
            _variant_investigation_apology(name),
            _variant_recollect(name, order_ref),
            _variant_acknowledge_ask_details(name, order_ref),
            _variant_apology_update(name),
        ]
    return [
        _variant_investigation_apology(name),
        _variant_thank_details(name),
        _variant_acknowledge_ask_details(name, order_ref),
        _variant_apology_update(name),
    ]


def _four_variants_follow_up(
    name: str, order_ref: str | None, *, service_failure: bool
) -> list[str]:
    _ = service_failure
    return [
        _variant_evidence_follow_up(name),
        _variant_investigation_update(name),
        _variant_recollect_confirm(name, order_ref),
        _variant_resolution_offer(name),
    ]


def suggest_complaint_reply(
    *,
    content_main: str,
    subject: str = "",
    from_header: str = "",
    thread: list[dict[str, Any]] | None = None,
    greeting_name: str = "there",
) -> dict:
    """Build four admin-style complaint reply options for staff to choose from."""
    _ = from_header
    name = greeting_name or "there"
    combined = f"{subject}\n{content_main}\n{thread_text(thread)}"
    customer_text = _customer_message_only(content_main)
    order_ref = _extract_order_ref(combined)
    follow_up = _is_follow_up(
        subject=subject,
        content_main=content_main,
        thread=thread,
    )
    service_failure = _is_service_failure_complaint(customer_text)

    if follow_up:
        templates = _four_variants_follow_up(name, order_ref, service_failure=service_failure)
        phase = "follow_up"
        reason = (
            "Four complaint follow-up options: photo acknowledged, investigation update, "
            "re-collection, resolution"
        )
    else:
        templates = _four_variants_first_contact(name, order_ref, service_failure=service_failure)
        phase = "acknowledge"
        reason = (
            "Four first-contact complaint options: investigation apology, "
            + (
                "re-collection, ask details, update"
                if service_failure
                else "thank details, ask details, update"
            )
        )

    return return_replies(
        {
            "category": "complaint",
            "phase": phase,
            "missing_slots": [],
            "reason": reason,
            "customer_name": name,
            "order_ref": order_ref,
            "is_service_failure": service_failure,
            "is_follow_up": follow_up,
            "language": "en-GB",
        },
        cap_suggestions(templates),
        draft_source="rules",
    )
