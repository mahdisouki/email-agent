"""
LLM helpers for classifier, quote replies, order extraction, and catalogue pricing.

Uses a self-hosted Ollama-compatible POST /api/generate endpoint.

# Groq (disabled — previously used for chat completions):
#   GROQ_API_KEY, GROQ_SUGGEST_MODEL, GROQ_CLASSIFY_MODEL
#   from groq import Groq; client.chat.completions.create(...)
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv

load_dotenv()

OLLAMA_GENERATE_URL = (
    os.getenv("OLLAMA_GENERATE_URL") or "https://olama.duckdns.org/api/generate"
).strip()
OLLAMA_MODEL = (os.getenv("OLLAMA_MODEL") or "qwen2.5:3b-instruct").strip()
SUGGEST_MODEL = OLLAMA_MODEL
USE_LLM_SUGGEST = os.getenv("USE_LLM_SUGGEST", "true").lower() in ("1", "true", "yes")
_LLM_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC") or "120")

# Groq — commented out; kept for reference only.
# GROQ_API_KEY = (os.getenv("GROQ_API_KEY") or "").strip()
# SUGGEST_MODEL = os.getenv("GROQ_SUGGEST_MODEL", "qwen/qwen3-32b")


def cap_suggestions(replies: list[str]) -> list[str]:
    """Return all non-empty suggestions unless MAX_REPLY_SUGGESTIONS caps the count."""
    cleaned = [r for r in replies if r]
    raw = (os.getenv("MAX_REPLY_SUGGESTIONS") or "").strip()
    if not raw or raw == "0":
        return cleaned
    try:
        limit = int(raw)
    except ValueError:
        return cleaned
    if limit < 1:
        return cleaned
    return cleaned[:limit]


MAX_SUGGESTIONS = 0


def is_llm_suggest_enabled() -> bool:
    return bool(OLLAMA_GENERATE_URL) and USE_LLM_SUGGEST


def ollama_generate(
    *,
    prompt: str,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    json_mode: bool = False,
) -> str:
    """Call Ollama POST /api/generate (non-streaming). Returns assistant text."""
    if not OLLAMA_GENERATE_URL:
        raise RuntimeError("OLLAMA_GENERATE_URL is not configured")

    full_prompt = (prompt or "").strip()
    if system and system.strip():
        full_prompt = f"{system.strip()}\n\n{full_prompt}"

    payload: dict[str, Any] = {
        "model": (model or OLLAMA_MODEL).strip(),
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if json_mode:
        payload["format"] = "json"

    req = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_LLM_TIMEOUT_SEC) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    if not isinstance(body, dict):
        raise RuntimeError("Ollama returned non-object JSON")

    text = body.get("response")
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError("Ollama returned empty response")


def llm_chat(
    *,
    system: str,
    user: str,
    temperature: float = 0.7,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """System + user prompt via Ollama /api/generate."""
    return ollama_generate(
        system=system,
        prompt=user,
        temperature=temperature,
        json_mode=json_mode,
        model=model,
    )


def extract_groq_message_text(message: Any) -> str:
    """Back-compat alias — accepts a raw string or legacy Groq message object."""
    if isinstance(message, str):
        return message.strip()
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    reasoning = getattr(message, "reasoning", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return ""


def groq_chat_extra_kwargs(model: str) -> dict:
    """No-op — Groq-only options (e.g. reasoning_effort) are not used with Ollama."""
    _ = model
    return {}


# Groq client (disabled):
# def _get_client():
#     global _client
#     if not GROQ_API_KEY:
#         return None
#     if _client is None:
#         from groq import Groq
#         _client = Groq(api_key=GROQ_API_KEY)
#     return _client


def _strip_staff_signature(text: str) -> str:
    """Keep only the body up to and including 'Kind regards.' — drop staff sign-off blocks."""
    lines = (text or "").splitlines()
    if not lines:
        return text
    out: list[str] = []
    for line in lines:
        out.append(line)
        if re.match(r"^\s*kind regards[,.]?\s*$", line.strip(), re.I):
            break
    body = "\n".join(out).rstrip()
    if body and not re.search(r"kind regards", body, re.I):
        body = f"{body.rstrip()}\n\nKind regards."
    return body


def llm_suggest_quote_companion(
    *,
    customer_name: str,
    phase: str,
    missing_slots: list[str],
    reason: str,
    subject: str,
    thread_summary: str,
    standard_replies: list[str],
    item_description: str = "",
    pricing_summary: str = "",
    availability_summary: str = "",
) -> tuple[str | None, str | None]:
    """
    Generate one additional reply variant to help move the quote conversation forward.
    Returns (reply_text, error).
    """
    if not is_llm_suggest_enabled():
        return None, "LLM disabled"

    standards = "\n\n---\n\n".join(
        f"Standard suggestion {i + 1}:\n{r[:3500]}"
        for i, r in enumerate(standard_replies[:4])
        if r
    )
    missing_text = ", ".join(missing_slots) if missing_slots else "none"
    prompt = f"""You draft customer service emails for London Waste Management (UK waste collection).

Write ONE additional reply email — complementary to the standard suggestion(s) below.
Cover different topics or ask for different details; do NOT repeat the same asks or paragraphs.

Rules:
- British English (UK spelling, floor numbering).
- Start with "Hi {customer_name}," then a blank line.
- End with "Kind regards." only — do NOT add your name, job title, company name, phone, email, or office address.
- Do NOT repeat topics already covered in STANDARD SUGGESTION(S): price, earliest collection date/time slots, photo requests, address/phone/email asks, or working hours — unless the standard suggestion omitted them entirely.
- Focus on MISSING SLOTS not already addressed in the standard suggestion(s), e.g. collection location (inside/outside, floor, lift), item weight, lift/crew requirements, dismantling.
- Do NOT invent prices unless they appear in PRICING below.
- Use ONLY dates and time slots listed in AVAILABILITY below — never invent times (e.g. do not say "10am onwards").
- If AVAILABILITY is empty or "(none yet)", do NOT invent dates or slots.
- NEVER repeat or echo the customer's questions back to them.
- If the customer asked for the latest slot, offer afternoon (12pm–5pm) or AnyTime from AVAILABILITY only.
- If mentioning working days, say we work Monday to Sunday (not Saturday only, no Sunday surcharge).
- Ask only for information listed in MISSING SLOTS (if any) that is not already asked in the standard suggestion(s).
- Be warm, professional, concise.
- Output ONLY the email body text — no labels, no markdown fences.

CONTEXT
Phase: {phase}
Reason: {reason}
Subject: {subject or "(none)"}
Missing slots: {missing_text}
Item description so far: {item_description or "(none yet)"}
PRICING: {pricing_summary or "(none yet)"}
AVAILABILITY (use these dates/slots only): {availability_summary or "(none yet)"}

THREAD (oldest first):
{thread_summary[:10000]}

STANDARD SUGGESTION(S) ALREADY PROVIDED (do NOT duplicate these topics — cover what they left out):
{standards[:12000] or "(none)"}
"""

    try:
        raw = llm_chat(
            system=(
                "You write UK waste-collection customer emails. "
                "Output only the email body."
            ),
            user=prompt,
            temperature=0.7,
        )
    except Exception as exc:
        return None, f"LLM request failed: {exc}"

    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("text"):
            text = text[4:].strip()
    if not text or len(text) < 40:
        return None, "LLM returned empty companion reply"
    if f"hi {customer_name.lower()}" not in text.lower()[:80]:
        text = f"Hi {customer_name},\n\n{text}"
    if "kind regards" not in text.lower():
        text = f"{text.rstrip()}\n\nKind regards."
    return _strip_staff_signature(text), None
