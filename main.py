import json
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request

load_dotenv()
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from core.gmail_message import messages_to_thread, normalize_message, pick_target_message
from models.schemas import ClassifyEmailRequest, OrderLLMRequest, SuggestReplyRequest, ThreadMessage
from services.classifier import classify_email
from services.customer_extract import extract_customer_from_conversation
from services.reply_suggester import suggest_reply

app = FastAPI(title="London Waste Management Email MCP")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LogPostBodyMiddleware(BaseHTTPMiddleware):
    """Keep raw POST body on request.state for 422 debugging."""

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            body = await request.body()

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request._receive = receive
            try:
                request.state.raw_body = json.loads(body.decode("utf-8")) if body else None
            except (json.JSONDecodeError, UnicodeDecodeError):
                request.state.raw_body = body.decode("utf-8", errors="replace")[:4000]

        return await call_next(request)


app.add_middleware(LogPostBodyMiddleware)


def _log(title: str, payload) -> None:
    print("\n" + "=" * 60)
    print(title)
    if isinstance(payload, (dict, list)):
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(payload)
    print("=" * 60 + "\n")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raw = getattr(request.state, "raw_body", None)
    _log(
        f"❌ 422 Unprocessable Entity — {request.method} {request.url.path}",
        {
            "validation_errors": exc.errors(),
            "received_body": raw,
            "expected_shape": {
                "success": True,
                "message": {
                    "contentMain": "required (plain text)",
                    "from": "sender",
                    "subject": "optional",
                    "aiCategory": "quote | complaint | ...",
                },
                "thread": "optional array",
            },
            "hint": "Do not send flat { subject, contentMain } — wrap inside message.",
        },
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


def _require_content(content: str, field: str = "contentMain") -> str:
    text = (content or "").strip()
    if not text:
        raise HTTPException(status_code=422, detail=f"{field} is required")
    return text


def _normalize_thread(thread: list[ThreadMessage] | None) -> list[dict] | None:
    if not thread:
        return None
    out = []
    for m in thread:
        row = m.model_dump(by_alias=True)
        from_raw = row.pop("from_address", None) or row.get("from") or ""
        explicit_dir = (row.get("direction") or "").strip().lower()
        norm = normalize_message(
            {
                "from": from_raw,
                "to": row.get("to"),
                "subject": row.get("subject"),
                "contentMain": row.get("contentMain"),
                "snippet": row.get("snippet"),
            }
        )
        if explicit_dir in ("inbound", "outbound"):
            row["direction"] = explicit_dir
        else:
            row["direction"] = norm["direction"]
        row["from"] = norm["from"]
        out.append(row)
    return out


@app.get("/health")
def health():
    return {"status": "healthy"}


def _classify(data: ClassifyEmailRequest) -> dict:
    content = _require_content(data.contentMain)
    _log("📥 Classify — request", {
        "subject": data.subject,
        "contentMain": content,
        "snippet": data.snippet,
        "from_address": data.from_address,
        "messages_count": len(data.messages) if data.messages else 0,
    })
    result = classify_email(
        data.subject,
        content,
        snippet=data.snippet or "",
        from_address=data.from_address or "",
        messages=data.messages,
    )
    _log("📧 Classify — response", result)
    return result


@app.post("/tools/classify_email")
def classify_email_route(data: ClassifyEmailRequest):
    return _classify(data)


def _suggest_reply(data: SuggestReplyRequest) -> dict:
    if not data.message:
        raise HTTPException(
            status_code=422,
            detail="message is required (wrap contentMain inside message: { ... })",
        )
    latest = normalize_message(data.message.model_dump(by_alias=True))
    thread_from_messages = messages_to_thread(data.messages)
    thread = thread_from_messages or _normalize_thread(data.thread)
    target, warning = pick_target_message(latest, thread)
    body = (target.get("contentMain") or target.get("snippet") or "").strip()
    if not body:
        raise HTTPException(
            status_code=422,
            detail="message.contentMain or message.snippet is required",
        )
    target = {**target, "contentMain": body}

    _log("📥 Suggest reply — request", {
        "message": target,
        "thread": thread,
    })

    attachments = target.get("attachments")
    if attachments is None and data.message is not None:
        attachments = data.message.attachments

    result = suggest_reply(
        subject=target["subject"],
        content_main=target["contentMain"],
        snippet=target.get("snippet") or "",
        from_header=target.get("from") or "",
        direction=target.get("direction") or "inbound",
        thread=thread,
        parsed_form=None,
        attachments=attachments,
        category=target.get("aiCategory"),
    )

    result["threadId"] = (
        data.threadId
        or target.get("threadId")
        or latest.get("threadId")
        or (data.message.threadId if data.message else None)
    )
    result["messageId"] = (
        data.messageId
        or target.get("messageId")
        or target.get("id")
        or (data.message.messageId if data.message else None)
        or (data.message.id if data.message else None)
    )
    if warning:
        result["warning"] = warning

    _log("✉️ Suggest reply — response", result)
    return result


@app.post("/suggest_reply")
def suggest_reply_route(data: SuggestReplyRequest):
    """Draft LWM reply from your backend latest-message payload."""
    return _suggest_reply(data)


def _order_llm(data: OrderLLMRequest) -> dict:
    if not data.message:
        raise HTTPException(
            status_code=422,
            detail="message is required (wrap contentMain inside message: { ... })",
        )
    latest = normalize_message(data.message.model_dump(by_alias=True))
    thread_from_messages = messages_to_thread(data.messages)
    thread = thread_from_messages or _normalize_thread(data.thread)
    target, warning = pick_target_message(latest, thread)
    body = (target.get("contentMain") or target.get("snippet") or "").strip()
    if not body and not thread:
        raise HTTPException(
            status_code=422,
            detail="message.contentMain or thread is required",
        )

    _log("📥 Order LLM — request", {
        "message": target,
        "thread": thread,
    })

    result = extract_customer_from_conversation(
        thread=thread,
        content_main=body or (thread[-1].get("contentMain") if thread else ""),
        from_header=target.get("from") or "",
        subject=target.get("subject") or "",
    )

    result["threadId"] = (
        data.threadId
        or target.get("threadId")
        or latest.get("threadId")
        or (data.message.threadId if data.message else None)
    )
    result["messageId"] = (
        data.messageId
        or target.get("messageId")
        or target.get("id")
        or (data.message.messageId if data.message else None)
        or (data.message.id if data.message else None)
    )
    if warning:
        result["warning"] = warning

    if result.get("llm_error"):
        raise HTTPException(status_code=503, detail=result["llm_error"])

    _log("👤 Order LLM — response", result)
    return result


@app.post("/order_llm")
def order_llm_route(data: OrderLLMRequest):
    """Extract customer contact details and items from a full conversation thread."""
    return _order_llm(data)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
