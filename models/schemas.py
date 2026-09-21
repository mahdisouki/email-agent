from pydantic import BaseModel, Field, model_validator


class ClassifyEmailRequest(BaseModel):
    subject: str = ""
    contentMain: str
    snippet: str | None = None
    from_address: str | None = None
    """Optional thread messages for context (subject, snippet, from, aiCategory per row)."""
    messages: list[dict] | None = None


class ThreadMessage(BaseModel):
    direction: str = "inbound"
    subject: str = ""
    contentMain: str = ""
    from_address: str | None = Field(None, alias="from")
    to: str | None = None
    sent_at: str | None = None

    model_config = {"populate_by_name": True}


class PricingHints(BaseModel):
    """Optional on request — always returned empty; staff fills price in the app."""

    ballpark_gbp: int | None = None
    final_gbp: int | None = None
    availability: str | None = None


class GmailMessage(BaseModel):
    """Matches your backend latest-message `message` object."""

    id: str | None = None
    messageId: str | None = None
    threadId: str | None = None
    from_address: str | None = Field(None, alias="from")
    to: str | None = None
    subject: str = ""
    contentMain: str = ""
    snippet: str | None = None
    aiCategory: str | None = None
    date: str | None = None
    labelIds: list[str] | None = None
    internalDate: str | None = None
    attachments: list[dict] | None = None

    model_config = {"populate_by_name": True}


class DetectedItem(BaseModel):
    """Vision/upload-detected item from the client (name + count)."""

    name: str
    count: int = 1


class SuggestReplyRequest(BaseModel):
    """POST your backend `{ success, message }` response as-is."""

    success: bool = True
    message: GmailMessage | None = None
    messages: list[dict] | None = None
    thread: list[ThreadMessage] | None = None
    pricing: PricingHints | None = None
    detectedItems: list[DetectedItem] | None = None
    threadId: str | None = None
    messageId: str | None = None
    # Top-level category from Node (fallback when thread target has no aiCategory)
    aiCategory: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data):
        if not isinstance(data, dict):
            return data

        def _coalesce_id(raw: dict) -> str | None:
            mid = raw.get("messageId") or raw.get("message_id")
            iid = raw.get("id")
            return str(mid or iid).strip() if (mid or iid) else None

        def _merge_ids_into_message(msg: dict, outer: dict) -> dict:
            if not isinstance(msg, dict):
                return msg
            mid = _coalesce_id(msg) or _coalesce_id(outer)
            tid = msg.get("threadId") or outer.get("threadId") or outer.get("thread_id")
            out = {**msg}
            if mid:
                out.setdefault("id", mid)
                out.setdefault("messageId", mid)
            if tid:
                out.setdefault("threadId", tid)
            # Carry outer category onto message when message lacks one
            outer_cat = outer.get("aiCategory") or outer.get("category")
            if outer_cat and not (out.get("aiCategory") or out.get("category")):
                out["aiCategory"] = outer_cat
            return out

        # Backend thread array: use last as message if message omitted
        if data.get("messages") and not data.get("message"):
            msgs = data["messages"]
            if isinstance(msgs, list) and msgs:
                data = {**data, "message": msgs[-1]}

        # Preserve top-level category alias
        top_cat = data.get("aiCategory") or data.get("category")

        if data.get("message") is not None:
            msg = _merge_ids_into_message(data["message"], data)
            mid = _coalesce_id(data) or _coalesce_id(msg)
            tid = data.get("threadId") or msg.get("threadId")
            out = {
                **data,
                "message": msg,
                **({"messageId": mid} if mid else {}),
                **({"threadId": tid} if tid else {}),
            }
            if top_cat:
                out["aiCategory"] = top_cat
            return out

        # Flat body (Node suggestReplyService: threadId, messageId, contentMain, …)
        if data.get("contentMain") is not None:
            mid = _coalesce_id(data)
            tid = data.get("threadId") or data.get("thread_id")
            out = {
                "success": data.get("success", True),
                "messageId": mid,
                "threadId": tid,
                "message": {
                    "id": mid,
                    "messageId": mid,
                    "threadId": tid,
                    "from": data.get("from") or data.get("from_address"),
                    "to": data.get("to"),
                    "subject": data.get("subject") or "",
                    "contentMain": data.get("contentMain"),
                    "snippet": data.get("snippet"),
                    "aiCategory": data.get("aiCategory") or data.get("category"),
                    "date": data.get("date"),
                },
                "thread": data.get("thread"),
                "detectedItems": data.get("detectedItems") or data.get("detected_items"),
            }
            if data.get("items") is not None:
                out["items"] = data.get("items")
            if top_cat:
                out["aiCategory"] = top_cat
            return out
        return data


class CustomerDetails(BaseModel):
    firstName: str | None = None
    lastName: str | None = None
    phoneNumber: str | None = None
    email: str | None = None
    postcode: str | None = None
    address: str | None = None


class ExtractedItem(BaseModel):
    phrase: str
    quantity: int = 1
    status: str = "custom"
    item_id: str | None = None
    item_name: str | None = None
    type: str | None = None  # optional alias from frontend (standard | custom)
    price: float | None = None
    inside: float | None = None
    inside_with_dismantling: float | None = None
    final_price: float | None = None
    # Outside | Inside | Inside with dismantling
    position: str | None = None


class OrderLLMRequest(BaseModel):
    """Same envelope as suggest_reply — message + optional thread + items from frontend."""

    success: bool = True
    message: GmailMessage | None = None
    messages: list[dict] | None = None
    thread: list[ThreadMessage] | None = None
    threadId: str | None = None
    messageId: str | None = None
    # Items come from the frontend (e.g. quote suggest_reply) — not guessed by LLM
    items: list[dict] | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_payload(cls, data):
        if not isinstance(data, dict):
            return data
        items = data.get("items")
        normalized = SuggestReplyRequest.normalize_payload(data)
        if isinstance(normalized, dict) and items is not None and "items" not in normalized:
            normalized = {**normalized, "items": items}
        elif isinstance(normalized, dict) and items is not None:
            normalized.setdefault("items", items)
        return normalized


class OrderLLMResponse(BaseModel):
    customer: CustomerDetails
    items: list[ExtractedItem]
    source: str
    bookingDate: str | None = None
    bookingTimeSlot: str | None = None
    customerNote: str | None = None
    # Outside | Inside | Inside with dismantling
    position: str | None = None
    threadId: str | None = None
    messageId: str | None = None
    subject: str | None = None
    llm_error: str | None = None
