"""
LLM helpers for classifier, order extraction, catalogue item extraction, and pricing.

Uses a self-hosted Ollama-compatible POST /api/generate endpoint.

# Groq (disabled — previously used for chat completions):
#   GROQ_API_KEY, GROQ_SUGGEST_MODEL, GROQ_CLASSIFY_MODEL
#   from groq import Groq; client.chat.completions.create(...)
"""
from __future__ import annotations

import json
import os
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
