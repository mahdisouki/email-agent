import json
import os
import re
from typing import Any

from dotenv import load_dotenv

from core.llm_service import OLLAMA_MODEL, is_llm_suggest_enabled, llm_chat

load_dotenv()

# Groq (disabled):
# GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
# CLASSIFY_MODEL = os.getenv("GROQ_CLASSIFY_MODEL", "qwen/qwen3-32b")
CLASSIFY_MODEL = OLLAMA_MODEL
USE_LLM_CLASSIFY = os.getenv("USE_LLM_CLASSIFY", "true").lower() in ("1", "true", "yes")

# _classify_client = None  # Groq client — no longer used

VALID_CATEGORIES = {
    "quote",
    "complaint",
    "order_confirmation",
    "update_order",
    "invoices",
    "careers",
    "moving_services",
    "other",
}

# Used only when rules did NOT match — LLM recheck before returning other
LLM_RECHECK_SYSTEM_PROMPT = """You recheck emails for London Waste Management that did NOT match our rule-based classifier.
Rules already handled obvious cases (payment confirmations, supplier invoices, careers, moving requests,
standard quotation forms, clear complaints, order updates). Your job: pick the best category — especially quote vs other.

Categories (same as production):
- quote: Price/collection/removal request OR quote-thread follow-up. Includes:
  * "New Contact Request" forms: First Name, Email, Phone, Subject/Message (e.g. TV collection, "can you help me").
  * "New Quotation Request" with Comments.
  * Customer follow-ups: photos attached, "more to follow", "re our email yesterday", subject "Collection ...",
    sending pictures for access/items — NOT a complaint unless clear dissatisfaction about completed work.
- complaint: Unhappy about completed service, formal complaint, mess left, #ORD + poor service tone.
- order_confirmation, update_order, invoices, careers, moving_services: only if clearly fits despite rules missing it.
- other: Marketing, spam, unrelated general enquiry only.

Examples (classify as quote):
Email: "New Contact Request ... First Name: Lesley Ann ... Subject: TV collection Message: I need a TV collection, can you help me"
→ {"category":"quote","confidence":0.96,"reason":"Contact form requesting TV collection"}

Email subject: "Collection SE13"
Body: "Hi here are some photos re our email yesterday. more to follow"
→ {"category":"quote","confidence":0.93,"reason":"Photo follow-up in collection/quote thread"}

Email: "New Quotation Request\\nFirst Name: VASSILI\\nComments: Builders mixed waste"
→ {"category":"quote","confidence":0.98,"reason":"Structured quotation request"}

Examples (not quote):
Email: "What are your opening hours on Saturday?" → other
Email: "#ORD037736 formal complaint sawdust left" → complaint

Use thread context when provided. Photos for quoting/access → quote. Return ONLY JSON: category, confidence, reason."""


def _subject_indicates_quote_thread(subject: str) -> bool:
    s = subject.lower()
    return "quotation request" in s or "new quotation" in s


def _indicates_quote(subject: str, contentMain: str, snippet: str = "") -> bool:
    """New quotation requests and quotation-thread replies (not order updates)."""
    if _indicates_update_order(subject, contentMain, snippet):
        return False
    if _subject_indicates_quote_thread(subject):
        return True
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    if "new quotation request" in text:
        return True
    if (
        "first name:" in text
        and "comments:" in text
        and ("email:" in text or "phone number:" in text)
    ):
        return True
    if "new contact request" in text and "first name:" in text:
        if ("message:" in text or "subject:" in text) and (
            "email:" in text or "phone" in text
        ):
            return True
    return False


def _indicates_update_order(subject: str, body: str, snippet: str = "") -> bool:
    """
    Order/quote updates: customer or staff changing an existing booking — with or without #ORD.
    Covers quote changes, reschedules, and dispatch collection time-slot updates.
    """
    text = f"{subject}\n{snippet}\n{body}".lower()
    subject_lower = (subject or "").lower()

    # Subject-led (no order number required)
    if "collection update" in subject_lower or "order update" in subject_lower:
        return True

    # Order reference + explicit change request (original rule)
    has_order_ref = "#ord" in text or re.search(r"\bord\d{5,}\b", text) is not None
    change_markers = (
        "quote change",
        "change request",
        "update my quote",
        "update my order",
        "update the quote",
        "update the order",
        "change my quote",
        "change my order",
        "affects the price",
        "updated payment",
        "revised quote",
        "modify my",
        "amend my",
        "reschedule",
        "re-schedule",
        "change the date",
        "change the time",
        "change my collection",
        "move my collection",
    )
    if has_order_ref and any(m in text for m in change_markers):
        return True

    # Dispatch / schedule updates (no order number required)
    strong_dispatch_markers = (
        "updated time slot",
        "update time slot",
        "last-minute change",
        "dispatching team has informed",
        "dispatching team has confirmed",
        "dispatching team informed",
        "collection time for today has been adjusted",
        "collection time has been adjusted",
        "collection time has been changed",
        "your updated time slot",
    )
    if any(m in text for m in strong_dispatch_markers):
        return True

    schedule_markers = (
        "updated time slot",
        "collection time",
        "dispatching team",
        "last-minute change",
        "12pm-5pm",
        "7am-12pm",
        "collection date",
        "collection day",
        "time slot",
    )
    change_context = (
        "adjusted",
        "changed",
        "updated",
        "reschedul",
        "maintenance",
        "notice prior",
        "informed us",
        "last-minute",
    )
    if any(m in text for m in schedule_markers) and any(c in text for c in change_context):
        return True

    return False


def _indicates_invoices(subject: str, contentMain: str, snippet: str = "") -> bool:
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    if any(
        x in text
        for x in (
            "payment confirmation",
            "successfully received your payment for order",
            "thank you for choosing london waste management",
        )
    ):
        return False
    if re.search(r"\binvoice\s+\d+", text) and "payment confirmation" not in text:
        return True
    markers = (
        "e-billing",
        "e-billing document",
        "monthly invoicing",
        "invoice(s) from",
        "invoices from",
        "invoice no.",
        "invoice no ",
        "review and pay",
        "powered by quickbooks",
        "remittance advices",
        "credit@powerday",
        "ebill.suez",
        "suez group",
        "suez recycling",
        "powerday plc",
        "chatham freight",
        "documents from suez",
        "your latest documents from",
    )
    return any(m in text for m in markers)


def _indicates_careers(subject: str, contentMain: str, snippet: str = "") -> bool:
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    markers = (
        "recruitment application",
        "new recruitment application",
        "job application",
        "submitted a job application",
        "role of interest",
        "experience level",
        "cv uploaded",
        "view cv",
        "careers@",
        "job opening",
        "apply for the position",
        "application for the role",
    )
    return any(m in text for m in markers)


def _indicates_complaint(subject: str, contentMain: str, snippet: str = "") -> bool:
    if _indicates_update_order(subject, contentMain, snippet):
        return False
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    markers = (
        "formal complaint",
        "make a complaint",
        "would like to complain",
        "file a complaint",
        "not acceptable",
        "very disappointed",
        "poor service",
        "left behind",
        "sawdust",
        "rubbish left",
    )
    has_complaint_word = "complaint" in text or "complain" in text
    return has_complaint_word and any(m in text for m in markers) or (
        "complaint" in text and ("#ord" in text or re.search(r"\bord\d{5,}\b", text))
    )


def _indicates_moving_services(subject: str, contentMain: str, snippet: str = "") -> bool:
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    markers = (
        "moving service request",
        "new moving service request",
        "submitted a moving service request",
        "pick up location",
        "drop off location",
        "pick up property type",
        "drop off property type",
        "packing required",
        "access info:",
        "moving service",
        "home removal",
        "removals request",
    )
    return any(m in text for m in markers)


def _indicates_order_confirmation(subject: str, contentMain: str, snippet: str = "") -> bool:
    text = f"{subject}\n{snippet}\n{contentMain}".lower()
    if _indicates_invoices(subject, contentMain, snippet):
        return False
    markers = (
        "payment confirmation",
        "order confirmation",
        "successfully received your payment",
        "payment failed",
        "payment could not be completed",
        "could not be completed",
        "checkout session expired",
        "no money has been taken",
    )
    if any(m in text for m in markers):
        has_order_ref = "#ord" in text or re.search(r"\bord\d{5,}\b", text) is not None
        if has_order_ref or "payment confirmation" in text or "order confirmation" in text:
            return True
        if "payment failed" in text or "checkout session expired" in text:
            return True
    return False


def _classify_by_rules(
    subject: str,
    body: str,
    snippet: str = "",
) -> dict | None:
    """
    Run rule chain. Returns a result dict if a rule matched, None if no rule matched
    (caller should LLM-recheck before returning other).
    """
    if _indicates_order_confirmation(subject, body, snippet):
        return {
            "category": "order_confirmation",
            "confidence": 0.96,
            "reason": "Payment or order confirmation (subject/body/snippet)",
        }

    if _indicates_invoices(subject, body, snippet):
        return {
            "category": "invoices",
            "confidence": 0.95,
            "reason": "Supplier or vendor invoice / e-billing to London Waste Management",
        }

    if _indicates_careers(subject, body, snippet):
        return {
            "category": "careers",
            "confidence": 0.95,
            "reason": "Recruitment or job application email",
        }

    if _indicates_moving_services(subject, body, snippet):
        return {
            "category": "moving_services",
            "confidence": 0.95,
            "reason": "Customer moving service request",
        }

    if _indicates_complaint(subject, body, snippet):
        return {
            "category": "complaint",
            "confidence": 0.94,
            "reason": "Customer complaint about service",
        }

    if _indicates_update_order(subject, body, snippet):
        return {
            "category": "update_order",
            "confidence": 0.94,
            "reason": "Order or collection schedule/quote update (with or without order reference)",
        }

    if _indicates_quote(subject, body, snippet):
        return {
            "category": "quote",
            "confidence": 0.95,
            "reason": "Quotation request or quotation thread",
        }

    return None


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _normalize_llm_result(data: dict) -> dict:
    category = str(data.get("category", "other")).lower().strip().replace(" ", "_")
    if category in ("order_confirmation", "orderconfirm", "payment_confirmation"):
        category = "order_confirmation"
    if category in ("update_order", "update_orders", "order_update", "quote_change"):
        category = "update_order"
    if category in ("invoices", "invoice", "supplier_invoice", "vendor_invoice"):
        category = "invoices"
    if category in ("careers", "career", "recruitment", "hr", "job_application"):
        category = "careers"
    if category in ("moving_services", "moving_service", "moving", "removals"):
        category = "moving_services"
    if category not in VALID_CATEGORIES:
        category = "other"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(data.get("reason", "")).strip() or "No reason provided"
    if not reason.lower().startswith("llm recheck"):
        reason = f"LLM recheck (no rule match): {reason}"

    return {
        "category": category,
        "confidence": confidence,
        "reason": reason,
    }


def _format_thread_for_llm(messages: list[dict[str, Any]] | None, *, max_messages: int = 12) -> str:
    if not messages:
        return ""
    rows = messages[-max_messages:] if len(messages) > max_messages else messages
    lines = []
    for i, m in enumerate(rows, 1):
        if not isinstance(m, dict):
            continue
        from_raw = m.get("from") or m.get("from_address") or "(unknown)"
        subj = (m.get("subject") or "").strip() or "(no subject)"
        snip = (m.get("snippet") or m.get("contentMain") or "").strip()
        prior = m.get("aiCategory") or m.get("ai_category") or ""
        lines.append(
            f"  [{i}] From: {from_raw}\n"
            f"      Subject: {subj}\n"
            f"      Snippet: {snip[:500]}\n"
            f"      Prior category: {prior or 'none'}"
        )
    if not lines:
        return ""
    return "\n\nThread context:\n" + "\n".join(lines)


def _classify_with_llm_recheck(
    subject: str,
    body: str,
    snippet: str,
    from_address: str,
    messages: list[dict[str, Any]] | None = None,
) -> dict | None:
    """LLM recheck when no rule matched. Returns None if unavailable or on error."""
    if not USE_LLM_CLASSIFY or not is_llm_suggest_enabled():
        return None

    try:
        thread_block = _format_thread_for_llm(messages)
        user_prompt = f"""No rule matched this email. Recheck the category (focus on quote vs other).

From: {from_address or "(unknown)"}
Subject: {subject or "(no subject)"}
Snippet: {snippet[:800] if snippet else "(none)"}

Body:
{body[:12000]}
{thread_block}
"""

        raw = llm_chat(
            system=LLM_RECHECK_SYSTEM_PROMPT,
            user=user_prompt,
            temperature=0.0,
            json_mode=True,
            model=CLASSIFY_MODEL,
        )
        if not raw.strip():
            print(f"[classifier] LLM recheck empty content model={CLASSIFY_MODEL}")
            return None
        parsed = json.loads(_strip_json_fences(raw))
        return _normalize_llm_result(parsed)
    except Exception as exc:
        print(f"[classifier] LLM recheck failed model={CLASSIFY_MODEL}: {exc}")
        return None


def classify_email(
    subject: str,
    contentMain: str,
    snippet: str = "",
    from_address: str = "",
    messages: list[dict[str, Any]] | None = None,
) -> dict:
    """
    Classify an email: rules first; if no rule matches, LLM recheck before returning other.
    Returns: {"category": ..., "confidence": float, "reason": str}
    """
    subject = subject.strip()
    body = contentMain.strip()
    snippet = (snippet or "").strip()

    rule_result = _classify_by_rules(subject, body, snippet)
    if rule_result:
        return rule_result

    llm_result = _classify_with_llm_recheck(subject, body, snippet, from_address, messages)
    if llm_result:
        return llm_result

    return {
        "category": "other",
        "confidence": 0.5,
        "reason": "No matching rule; LLM recheck unavailable or failed",
    }
