"""
Quote suggest reply — intake, follow-up, weight/lift, and location for UK waste quotes.

All customer-facing copy uses British English (UK spelling, floor numbering, access terms).
Customers are UK-based and often describe access in informal British phrasing (alley, ginnel,
mews, flat, etc.).
"""
from __future__ import annotations

import hashlib
import re
from typing import Any

from dataclasses import dataclass

from core.gmail_message import is_lwm_address, parse_display_name, parse_email_address
from core.llm_service import cap_suggestions, is_llm_suggest_enabled, llm_suggest_quote_companion
from services.reply_suggester import field, parse_form_fields, return_replies
from services.standard_items import (
    apply_pricing_to_suggested_replies,
    llm_resolve_thread_item_context,
    lookup_prices_for_text,
)
from services.task_availability import (
    availability_metadata,
    build_booking_availability_plan,
    confirmation_offer_line,
    customer_asks_availability,
    customer_asks_slot_options,
    customer_confirmed_booking_slot,
    customer_states_preferred_date_or_slot,
)

# British English — used in templates and metadata for downstream LLM prompts.
BRITISH_ENGLISH_CONTEXT = (
    "London Waste Management customers are UK-based. They write in British English "
    "(e.g. flat, mobile, postcode, ground/first floor, lift, alley, ginnel, mews). "
    "Staff replies must use British English spelling and phrasing throughout."
)

STANDARD_SLOT_LABELS: dict[str, str] = {
    "first_name": "your first name",
    "last_name": "your last name",
    "address": "your full collection address",
    "email": "your email address",
    "phone": "your phone number",
    "comments": "a description of the items or a list of what needs collecting",
}

_COMMENTS_PLACEHOLDERS = frozenset(
    {
        "",
        "-",
        "n/a",
        "na",
        "none",
        "nil",
        "see attached",
        "see attachment",
        "see photos",
        "see photo",
        "photos attached",
        "photo attached",
        "attached",
        "as above",
        "no additional comments",
        "no additional comment",
        "no comments",
        "no comment",
    }
)


def _is_comments_placeholder(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    low = re.sub(r"\s+", " ", raw.lower()).strip(" .")
    if low in _COMMENTS_PLACEHOLDERS:
        return True
    if re.search(
        r"^(?:no(?:ne|t applicable)?|n/?a|nil|same as above|"
        r"no additional comments?|nothing(?: to add| else)?|"
        r"no comment(?:s)?(?: (?:provided|given|added|to add))?|"
        r"not applicable|does not apply|don't know|dont know)\.?$",
        low,
    ):
        return True
    return False


def _comments_text(text: str) -> str:
    raw = (field(text, "Comments") or "").strip()
    raw = re.sub(r"uploaded items\s*\(\s*\d+\s*\).*", "", raw, flags=re.I | re.S).strip()
    raw = re.sub(r"view image.*", "", raw, flags=re.I | re.S).strip()
    return raw.strip()


def _looks_like_item_list(text: str) -> bool:
    if not text:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        return True
    if re.search(r"[,;]\s*\w", text):
        return True
    if re.search(r"^\s*[-*•\d]+[\.)]?\s+\w", text, re.M):
        return True
    if re.search(r"\b(and|plus|also)\b", text, re.I) and len(text.split()) >= 6:
        return True
    return False


def _meaningful_comments(content_main: str, form: dict[str, str]) -> bool:
    comments = (form.get("comments") or _comments_text(content_main) or "").strip()
    if _is_comments_placeholder(comments):
        comments = ""
    if comments and _meaningful_item_description(comments):
        return True
    is_form = "first name:" in (content_main or "").lower() and "comments:" in (
        content_main or ""
    ).lower()
    if comments and is_form and not _is_comments_placeholder(comments):
        snippet = _clean_customer_item_snippet(comments)
        if (
            snippet
            and not _is_address_or_contact_only(snippet)
            and not _is_photo_caption_only(snippet)
            and (
                _meaningful_item_description(snippet)
                or _text_describes_items(snippet)
            )
        ):
            return True
    if is_form:
        return False
    body = _strip_form_boilerplate(content_main)
    return _meaningful_item_description(body)


def _strip_form_boilerplate(text: str) -> str:
    if not text:
        return ""
    low = text.lower()
    if "first name:" not in low:
        return text.strip()

    for label in (
        "First Name",
        "Last Name",
        "Address",
        "Email",
        "Phone Number",
        "Comments",
        "Uploaded Items",
    ):
        text = re.sub(rf"^{re.escape(label)}:\s*.*$", "", text, flags=re.I | re.M)
    text = re.sub(r"thank you for using.*", "", text, flags=re.I | re.S)
    text = re.sub(r"view image.*", "", text, flags=re.I | re.S)
    return text.strip()


_ITEM_KEYWORDS = frozenset(
    {
        "sofa", "mattress", "bed", "wardrobe", "fridge", "freezer", "washer", "dryer",
        "rubbish", "waste", "bag", "bags", "tiles", "rubble", "wood", "furniture",
        "desk", "table", "chair", "pram", "pushchair", "cabinet", "cooker", "oven",
        "dishwasher", "shed", "fence", "door", "window", "bath", "toilet", "sink",
        "bin", "boxes", "cardboard", "scrap", "metal", "garden", "soil", "bricks",
        "piano", "upright", "keyboard", "organ",
    }
)


def _strip_mobile_signature(text: str) -> str:
    return re.split(
        r"\bsent from (?:my )?(?:iphone|ipad|android|outlook|samsung|huawei)\b",
        text or "",
        maxsplit=1,
        flags=re.I,
    )[0].strip()


def _strip_email_client_footer(text: str) -> str:
    """Remove common mobile/webmail footers before parsing contact details."""
    t = _strip_mobile_signature(text or "").strip()
    for pattern in (
        r"\byahoo mail\b.*",
        r"\bgmail\b.*",
        r"\bgoogle mail\b.*",
        r"\boutlook for (?:ios|android|windows)\b.*",
        r"\bget outlook for\b.*",
        r"\bmail for (?:ios|android|windows)\b.*",
        r"\bsent from mail for\b.*",
        r"\bprotonmail\b.*",
        r"\bthunderbird\b.*",
    ):
        t = re.split(pattern, t, maxsplit=1, flags=re.I)[0].strip()
    t = re.sub(r"\b\d+\s+photos?\s+have\s+been\s+sent\.?\s*$", "", t, flags=re.I).strip()
    return t


_STREET_SUFFIX = (
    r"(?:\broad\b|\brd\b|\bstreet\b|\bst\b|\blane\b|\bln\b|\bave\b|\bavenue\b|\bway\b|"
    r"\bdrive\b|\bdr\b|\bclose\b|\bcrescent\b|\bgrove\b|\bhill\b|\bcourt\b|\bplace\b|"
    r"\bterrace\b|\bmews\b|\bwalk\b|\bsquare\b|\bsq\b|\bparade\b|\brow\b|\bestate\b|"
    r"\bboulevard\b|\bblvd\b|\bgardens\b|\bpark\b|\bpath\b|\bgreen\b|\brise\b|\bview\b)"
)


def _text_describes_items(text: str) -> bool:
    body = _strip_mobile_signature(_strip_customer_reply(text or "")).lower()
    if not body:
        return False
    if _looks_like_item_list(body):
        return True
    return any(re.search(rf"\b{re.escape(w)}\b", body) for w in _ITEM_KEYWORDS)


def _item_keyword_hits(text: str) -> list[str]:
    body = (text or "").lower()
    hits: list[str] = []
    for w in _ITEM_KEYWORDS:
        if re.search(rf"\b{re.escape(w)}s?\b", body):
            hits.append(w)
    return hits


def _has_explicit_item_list(text: str) -> bool:
    """Inventory-style list — not incidental 'and' in prose (e.g. 'BBQ and not the tiles')."""
    body = (text or "").lower()
    if re.search(r"\bcontent in the pictures?\b", body):
        return True
    if re.search(r"\bconsisting of\b", body):
        return True
    if re.search(
        r"\b\d+\s+(?:boxes?|bags?|items?|panels?|chairs?|cushions?|tiles?)\b", body
    ):
        return True
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if len(lines) >= 2:
        return True
    if re.search(r"^\s*[-*•\d]+[\.)]?\s+\w", text or "", re.M):
        return True
    item_hits = _item_keyword_hits(body)
    if len(item_hits) >= 3:
        return True
    if len(item_hits) >= 2 and re.search(r",\s*\w", body):
        return True
    if len(item_hits) >= 2 and re.search(
        r"\b(?:and|plus)\s+(?:some|a|an|\d+|\w+\s+(?:boxes?|bags?|panels?|cushions?))",
        body,
    ):
        return True
    return False


def _is_photo_caption_only(text: str) -> bool:
    """
    Photos sent with picture captions or clarifications but no real item list
    (e.g. "in the middle picture it is the tiles behind the BBQ, not the BBQ").
    """
    body = _clean_customer_item_snippet(text or "").lower()
    if not body:
        return False

    photo_cues = (
        "picture",
        "pictures",
        "photo",
        "photos",
        "attached",
        "below:",
        "middle picture",
        "first picture",
        "second picture",
        "see attached",
        "as shown",
        "in the image",
    )
    if not any(cue in body for cue in photo_cues):
        return False

    if _has_explicit_item_list(body):
        return False

    return True


def _meaningful_item_description(text: str) -> bool:
    snippet = _clean_customer_item_snippet(text or "")
    if not snippet or _is_address_or_contact_only(snippet):
        return False
    if _is_photo_caption_only(snippet):
        return False
    if _has_explicit_item_list(snippet):
        return True
    if _looks_like_item_list(snippet) and _text_describes_items(snippet):
        return True
    return _text_describes_items(snippet) and not _customer_sent_photos(snippet)


def _is_address_or_contact_only(text: str) -> bool:
    body = _strip_mobile_signature(_strip_customer_reply(text or "")).lower()
    if not body:
        return True
    if _looks_like_item_list(body):
        return False
    if any(re.search(rf"\b{re.escape(w)}\b", body) for w in _ITEM_KEYWORDS):
        return False
    has_addr = bool(
        re.search(r"\b\d+\s+\w+\s+(?:road|rd|street|st|lane|ave|way|drive|close)\b", body)
        or re.search(r"\b[a-z]{1,2}\d{1,2}[a-z]?\s*\d[a-z]{2}\b", body)
    )
    has_phone = bool(re.search(r"\b07\d{9}\b", body))
    word_count = len(body.split())
    if has_addr and word_count <= 35:
        return True
    if has_phone and not any(re.search(rf"\b{w}\b", body) for w in _ITEM_KEYWORDS):
        if word_count <= 20 or (has_addr and word_count <= 45):
            return True
    return False


def _score_item_snippet(snippet: str) -> tuple[int, int]:
    """Higher is better — based on item signals, not message position."""
    if _is_photo_caption_only(snippet):
        return (-100, 0)
    hits = len(_item_keyword_hits(snippet))
    score = hits
    if _has_explicit_item_list(snippet):
        score += 8
    if _looks_like_item_list(snippet):
        score += 3
    return (score, hits)


def _rules_pick_best_description(candidates: list[str]) -> str:
    """Pick the richest item description by content signals (not latest/longest)."""
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0][:500]
    ranked = sorted(
        ((snippet, *_score_item_snippet(snippet)) for snippet in candidates),
        key=lambda row: (row[1], row[2], len(row[0])),
        reverse=True,
    )
    best_score = ranked[0][1]
    if best_score < 1:
        return ""
    top = [row[0] for row in ranked if row[1] == best_score]
    return top[0][:500]


@dataclass
class ThreadItemContext:
    description: str = ""
    has_description: bool = False
    source: str = "none"


_THREAD_ITEM_CONTEXT_CACHE: dict[str, ThreadItemContext] = {}


def _thread_item_cache_key(customer_bodies: list[str]) -> str:
    return hashlib.sha256(
        "\x1f".join(b[:300] for b in customer_bodies).encode("utf-8")
    ).hexdigest()


def resolve_thread_item_context(
    form: dict[str, str],
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> ThreadItemContext:
    """
    Consolidate item descriptions across the thread — LLM when enabled, else
    content-scored rules (not latest/longest position rules).
    """
    customer_bodies = _customer_reply_bodies(thread, content_main, from_header)
    cache_key = _thread_item_cache_key(customer_bodies)
    if cache_key in _THREAD_ITEM_CONTEXT_CACHE:
        return _THREAD_ITEM_CONTEXT_CACHE[cache_key]

    form_body = _form_body_from_context(thread, content_main)
    structured_comments = (_comments_text(form_body) or "").strip()
    if (
        structured_comments
        and not _is_comments_placeholder(structured_comments)
        and _meaningful_item_description(structured_comments)
    ):
        ctx = ThreadItemContext(
            description=structured_comments[:500],
            has_description=True,
            source="form",
        )
        _THREAD_ITEM_CONTEXT_CACHE[cache_key] = ctx
        return ctx

    llm_result, _llm_err = llm_resolve_thread_item_context(customer_bodies)
    if llm_result and llm_result.get("has_item_description"):
        description = str(llm_result.get("description") or "").strip()
        if not description:
            items = llm_result.get("items") or []
            if items:
                description = ", ".join(
                    f"{i['quantity']}x {i['phrase']}" if i.get("quantity", 1) > 1 else i["phrase"]
                    for i in items
                    if isinstance(i, dict) and i.get("phrase")
                )
        if description:
            ctx = ThreadItemContext(
                description=description[:500],
                has_description=True,
                source="llm",
            )
            _THREAD_ITEM_CONTEXT_CACHE[cache_key] = ctx
            return ctx

    candidates: list[str] = []
    for body in customer_bodies:
        snippet = _clean_customer_item_snippet(body)
        if snippet and _meaningful_item_description(snippet):
            candidates.append(snippet)

    description = _rules_pick_best_description(candidates)
    ctx = ThreadItemContext(
        description=description,
        has_description=bool(description),
        source="rules" if description else "none",
    )
    _THREAD_ITEM_CONTEXT_CACHE[cache_key] = ctx
    return ctx


def _customer_sent_photos(body: str, attachments: list[dict[str, Any]] | None = None) -> bool:
    if attachments:
        for att in attachments:
            if not isinstance(att, dict):
                continue
            mime = (att.get("mimeType") or att.get("mimetype") or att.get("contentType") or "").lower()
            name = (att.get("filename") or att.get("name") or "").lower()
            if mime.startswith("image/") or re.search(r"\.(jpe?g|png|gif|webp|heic|bmp)$", name):
                return True

    text = _strip_mobile_signature(_strip_customer_reply(body or ""))
    if re.search(r"uploaded items\s*\(\s*([1-9]\d*)\s*\)", text, re.I):
        return True
    if re.search(r"\bview image\b", text, re.I):
        return True
    low = text.lower()
    if re.search(r"\.(jpe?g|png|gif|webp|heic)\b", low):
        return True
    if re.search(
        r"\b(here (?:are|is|'s)|please find|i(?:'ve| have) attached|attached (?:are|is)|photos attached)\b.{0,40}\b(photo|picture|image|pic)s?\b",
        low,
    ):
        return True
    if re.search(r"\b(photo|picture|image)s?\s+attached\b", low):
        return True
    # Yahoo / mobile clients: "1 photo has been sent", "2 photos have been sent"
    if re.search(r"\b\d+\s+photos?\s+have\s+been\s+sent\b", low):
        return True
    if re.search(r"\b(?:photos?|pictures?|images?)\s+have\s+been\s+sent\b", low):
        return True
    if re.search(r"\b(?:a\s+)?(?:photo|picture|image)\s+has\s+been\s+sent\b", low):
        return True
    if re.search(
        r"\b(?:i(?:'ve| have)|we(?:'ve| have))\s+sent\s+(?:\d+\s+)?(?:photos?|pictures?|images?)\b",
        low,
    ):
        return True
    if re.search(r"\b(?:see|with)\s+(?:the\s+)?attached\s+(?:photo|picture|image)s?\b", low):
        return True
    return False


def _has_customer_photos(
    content_main: str,
    attachments: list[dict[str, Any]] | None,
) -> bool:
    """Legacy wrapper — prefer customer-only bodies in _collect_slots."""
    return _customer_sent_photos(content_main, attachments)


def _resolve_email(form: dict[str, str], from_header: str) -> str | None:
    email = (form.get("email") or "").strip()
    if email:
        return email
    if is_lwm_address(from_header):
        return None
    addr = parse_email_address(from_header)
    return addr or None


def _missing_standard_fields(
    form: dict[str, str],
    content_main: str,
    from_header: str,
    *,
    has_photos: bool,
) -> list[str]:
    missing: list[str] = []
    if not (form.get("first_name") or "").strip():
        missing.append("first_name")
    if not (form.get("last_name") or "").strip():
        missing.append("last_name")
    if not (form.get("address") or "").strip():
        missing.append("address")
    if not _resolve_email(form, from_header):
        missing.append("email")
    if not (form.get("phone") or "").strip():
        missing.append("phone")

    has_description = _meaningful_comments(content_main, form)
    if not has_description and not has_photos:
        missing.append("comments")

    return missing


def quote_greeting_name(form: dict[str, str]) -> str | None:
    """First name if available, otherwise last name."""
    first = (form.get("first_name") or "").strip()
    if first:
        token = first.split()[0]
        if len(token) >= 2:
            return token.title() if not token.isupper() else token
    last = (form.get("last_name") or "").strip()
    if last:
        token = last.split()[0]
        if len(token) >= 2:
            return token.title() if not token.isupper() else token
    return None


def _join_missing_labels(slots: list[str]) -> str:
    labels = [STANDARD_SLOT_LABELS[s] for s in slots if s in STANDARD_SLOT_LABELS]
    if not labels:
        return "the remaining details"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" and {labels[-1]}"


def _variant_missing_fields(name: str, missing: list[str]) -> str:
    ask = _join_missing_labels(missing)
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        f"Could you please provide {ask}?\n\n"
        "Kind regards."
    )


def _variant_photos_need_list(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        "I appreciate the time you have taken to provide photos.\n\n"
        "Based on your pictures, I can give an approximate price of £[PRICE]+VAT.\n\n"
        "It would be amazing if you could give a list so I can ensure the price given is fair\n"
        "and accurate.\n\n"
        "I completely understand you might be busy so if it's difficult please do not worry,\n"
        "we can amend the price when our team arrives if necessary.\n\n"
        "Kind regards."
    )


def _appreciation_phrase(content_main: str, form: dict[str, str]) -> str:
    comments = (form.get("comments") or _comments_text(content_main) or "").strip()
    if _looks_like_item_list(comments):
        return "I appreciate the time you have taken to provide a list."
    return "I appreciate the time you have taken to provide a comment or message."


def _variant_list_need_photos(name: str, content_main: str, form: dict[str, str]) -> str:
    appreciation = _appreciation_phrase(content_main, form)
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        f"{appreciation}\n\n"
        "Based on your list, I can give an approximate price of £[PRICE]+VAT.\n\n"
        "It would be amazing if you could provide pictures so I can ensure the price given is\n"
        "completely fair and accurate.\n\n"
        "I completely understand you might be busy so if it's difficult please do not worry,\n"
        "we can amend the price when our team arrives if necessary.\n\n"
        "Kind regards."
    )


def _variant_unmatched_items_note(name: str, items: list[str]) -> str:
    if len(items) == 1:
        ref = items[0]
    else:
        ref = ", ".join(items[:-1]) + f" and {items[-1]}"
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        f"I noted {ref} on your list — could you please send a photo so I can match this "
        "to our price list accurately?\n\n"
        "Kind regards."
    )


def _list_need_photos_variants(
    name: str,
    content_main: str,
    form: dict[str, str],
    pricing: dict[str, Any] | None,
) -> list[str]:
    replies = [_variant_list_need_photos(name, content_main, form)]

    if pricing:
        unmatched = pricing.get("unmatched_items") or []
        if unmatched:
            replies.append(_variant_unmatched_items_note(name, unmatched))

    return cap_suggestions(replies)


def _analysis_text(content_main: str, form: dict[str, str]) -> str:
    comments = (form.get("comments") or _comments_text(content_main) or "").strip()
    address = (form.get("address") or field(content_main, "Address") or "").strip()
    parts = [p for p in (comments, address, content_main) if p]
    return "\n".join(parts)


@dataclass
class CollectionAccess:
    """What the customer said about where items are and how we reach them."""

    customer_phrase: str | None = None
    access_kind: str | None = None
    collection_side: str | None = None  # inside | outside | rear | garden | unknown

    has_floor: bool = False
    floor_label: str | None = None
    has_lift_info: bool = False
    lift_available: bool | None = None

    @property
    def outside(self) -> bool:
        return self.collection_side == "outside"

    @property
    def inside(self) -> bool:
        return self.collection_side in ("inside", "rear")

    @property
    def alley_inside(self) -> bool:
        """Backward-compatible flag: rear-access types that may imply inside collection."""
        return self.access_kind in (
            "ginnel",
            "snicket",
            "passage",
            "rear",
            "mews",
            "communal",
        )


_LOCATION_KEYWORDS = (
    "alley",
    "ginnel",
    "snicket",
    "passage",
    "mews",
    "garden",
    "shed",
    "garage",
    "drive",
    "driveway",
    "kerb",
    "kerbside",
    "pavement",
    "floor",
    "lift",
    "flat",
    "maisonette",
    "bungalow",
    "rear",
    "back",
    "porch",
    "hallway",
    "communal",
    "stairs",
    "loft",
    "cellar",
    "basement",
    "terrace",
    "block",
    "estate",
)


def _location_keyword_in_text(low: str, kw: str) -> bool:
    if kw == "lift":
        if re.search(r"\blift(?:able|ed|ing)\b", low):
            return False
        return bool(
            re.search(
                r"\b(?:lift available|with (?:a )?lift|no lift|passenger lift|stairs only)\b",
                low,
            )
        )
    return bool(re.search(rf"\b{re.escape(kw)}\b", low))


def _clean_location_phrase(phrase: str) -> str:
    phrase = re.sub(r"\s+", " ", (phrase or "").strip(" ,.;"))
    if not phrase:
        return phrase

    tail = re.search(r"\b(?:but|however|although)\s*(.+)$", phrase, re.I)
    if tail and any(_location_keyword_in_text(tail.group(1).lower(), kw) for kw in _LOCATION_KEYWORDS):
        phrase = tail.group(1).strip(" ,.;")

    m = re.search(
        r"\b("
        r"(?:this is|it's|its|access is|there is an?|down the|via the|through the|"
        r"round the back|at the back|in the|on the|everything is|items are|we have|"
        r"i'll leave|will leave|left on)"
        r".+)",
        phrase,
        re.I,
    )
    if m:
        phrase = m.group(1).strip(" ,.;")
    phrase = re.sub(r"^(?:hi[, ]*)?(?:location|access)\s*:\s*", "", phrase, flags=re.I).strip()
    return phrase


def _extract_location_phrase(text: str) -> str | None:
    """Pull the customer's own words about access/location (British English)."""
    body = _strip_mobile_signature(_strip_customer_reply(text or ""))
    if not body:
        return None

    for clause in re.split(r"[.!?\n]+", body):
        clause = re.sub(r"\s+", " ", clause).strip(" ,;")
        if not clause or len(clause) < 8:
            continue
        low = clause.lower()
        if re.search(r"\b(?:liftable|lifted|carrying|carried|one man|individual)\b", low):
            if not any(
                _location_keyword_in_text(low, kw)
                for kw in ("floor", "flat", "garden", "garage", "shed", "alley", "rear", "back")
            ):
                continue
        if not any(_location_keyword_in_text(low, kw) for kw in _LOCATION_KEYWORDS):
            continue
        if _is_address_or_contact_only(clause) and not any(
            _location_keyword_in_text(low, kw)
            for kw in (
                "alley",
                "ginnel",
                "garden",
                "floor",
                "lift",
                "flat",
                "rear",
                "back",
                "shed",
                "garage",
                "mews",
            )
        ):
            continue
        if len(clause) <= 140:
            return _clean_location_phrase(clause)
        m = re.search(
            r".{0,40}\b(?:"
            r"alley|ginnel|snicket|passage|mews|garden|shed|garage|drive(?:way)?|"
            r"rear|round the back|back of|floor|lift|flat|maisonette|bungalow|"
            r"communal|porch|hallway|loft|cellar|basement|kerb|pavement"
            r")\b.{0,60}",
            clause,
            re.I,
        )
        if m:
            return _clean_location_phrase(m.group(0).strip(" ,;."))
    return None


def _parse_floor_label(text: str) -> str | None:
    loc = re.search(r"\blocation\s*:\s*([^,.]+)", text or "", re.I)
    search_chunks = [loc.group(1)] if loc else []
    search_chunks.append(text or "")
    for chunk in search_chunks:
        nth = re.search(r"\b(\d+(?:st|nd|rd|th))\s+floor\b", chunk, re.I)
        if nth:
            return f"{nth.group(1).lower()} floor"
    patterns = (
        (r"\bground\s+floor\b|\bground\s+level\b|\bg\.?\s*f\.?\b", "ground floor"),
        (r"\blower\s+ground\b", "lower ground"),
        (r"\b(?:basement|cellar)\b", "basement"),
        (r"\b(?:first|1st)\s+floor\b", "first floor"),
        (r"\b(?:second|2nd)\s+floor\b", "second floor"),
        (r"\b(?:third|3rd)\s+floor\b", "third floor"),
        (r"\btop\s+floor\b", "top floor"),
        (r"\b(?:loft|attic)\b", "loft"),
        (r"\bmaisonette\b", "maisonette"),
        (r"\bbungalow\b", "bungalow"),
    )
    for pat, label in patterns:
        if re.search(pat, text or "", re.I):
            return label
    return None


def _parse_lift_state(text: str) -> tuple[bool, bool | None]:
    low = text.lower()
    if re.search(
        r"\b(?:no lift|no need for a lift|without a lift|there(?:'s| is) no lift|"
        r"lift (?:is )?broken|stairs only|walk[- ]up|no passenger lift|don't need a lift|"
        r"do not need a lift)\b",
        low,
    ):
        return True, False
    if re.search(
        r"\b(?:lift available|with a lift|with lift\b|there(?:'s| is) a lift|passenger lift|"
        r"service lift|we have a lift|lift from|lift access)\b",
        low,
    ):
        return True, True
    return False, None


def _match_access_kind(text: str) -> tuple[str | None, str | None]:
    """
    Return (access_kind, collection_side) from British access phrases.
    collection_side: inside | outside | rear | garden | unknown
    """
    rules: list[tuple[str, str, str]] = [
        (r"\bsnicket\b", "snicket", "rear"),
        (r"\b(?:ginnel|jitt(?:y|ey)|ten[- ]?foot|tenfoot)\b", "ginnel", "rear"),
        (r"\b(?:alley(?:way)?)\b", "alley", "outside"),
        (r"\b(?:down|via|through|access(?:ed)? by)\s+(?:an?\s+)?alley\b", "alley", "outside"),
        (r"\balley\s+(?:off|at|from|down)\b", "alley", "outside"),
        (r"\b(?:passage(?:way)?)\b", "passage", "rear"),
        (r"\b(?:down|via|through|access(?:ed)? by)\s+(?:an?\s+)?(?:passage|ginnel|snicket)\b", "passage", "rear"),
        (r"\b(?:round the back|round back|through the back|at the back|rear access|back entrance)\b", "rear", "rear"),
        (r"\brear of (?:the )?(?:property|house|flat|building)\b", "rear", "rear"),
        (r"\bmews\b", "mews", "rear"),
        (r"\b(?:communal (?:entrance|hall|hallway|stairs|area)|shared entrance)\b", "communal", "inside"),
        # Explicit inside
        (
            r"\b(?:just inside(?: the)?(?: front)? door|inside the front door|"
            r"from just inside|from inside the front door)\b",
            "flat",
            "inside",
        ),
        (r"\b(?:inside the (?:property|house|flat|premises)|from inside|in the house|in the flat|in the property)\b", "flat", "inside"),
        (r"\b(?:ground[- ]floor flat|first[- ]floor flat|top[- ]floor flat|basement flat)\b", "flat", "inside"),
        (r"\b(?:flat|maisonette|apartment)\b", "flat", "inside"),
        (r"\b(?:porch|hallway|hall)\b", "flat", "inside"),
        # Outside / external
        (r"\b(?:front|rear|back)\s+garden\b|\bin the garden\b|\bbottom of (?:the )?garden\b", "garden", "garden"),
        (r"\b(?:in the )?shed\b|\bgarden shed\b", "shed", "garden"),
        (r"\b(?:in the )?garage\b|\bdetached garage\b", "garage", "outside"),
        (r"\b(?:on|in) (?:the )?(?:drive(?:way)?|drive way)\b", "drive", "outside"),
        (r"\b(?:car park|parking bay|parking space)\b", "drive", "outside"),
        (r"\b(?:on the (?:street|road|pavement|kerb|curb)|kerbside|left outside|outside the (?:front|property|house))\b", "kerbside", "outside"),
        (r"\b(?:outside only|from outside|outside the property)\b", "kerbside", "outside"),
        (r"\bbungalow\b", "bungalow", "inside"),
    ]
    for pat, kind, side in rules:
        if re.search(pat, text, re.I):
            return kind, side
    return None, None


def _parse_collection_access(text: str, address: str = "") -> CollectionAccess:
    combined = f"{text}\n{address}"
    low = combined.lower()
    access = CollectionAccess()

    access.customer_phrase = _extract_location_phrase(combined) or _extract_location_phrase(address)
    access.access_kind, access.collection_side = _match_access_kind(low)

    access.floor_label = _parse_floor_label(low)
    access.has_floor = access.floor_label is not None
    if access.floor_label in ("bungalow", "ground floor", "maisonette"):
        access.has_floor = True

    access.has_lift_info, access.lift_available = _parse_lift_state(low)

    if access.floor_label == "bungalow":
        access.has_lift_info = True
        access.lift_available = False

    # Flat/maisonette/communal without floor — still need floor/lift unless stated
    if re.search(r"\b(?:flat|maisonette|apartment)\b", low) and not access.has_floor:
        access.collection_side = access.collection_side or "inside"
        access.access_kind = access.access_kind or "flat"

    if re.search(r"\b(?:end of terrace|mid terrace|terraced house|terrace house)\b", low):
        access.access_kind = access.access_kind or "terrace"

    return access


def _collection_location_complete(access: CollectionAccess) -> bool:
    # Clear external collection points
    if access.collection_side in ("outside", "garden") and access.access_kind in (
        "garden",
        "shed",
        "garage",
        "drive",
        "kerbside",
        "alley",
    ):
        return True

    if access.access_kind in ("kerbside", "alley") and not access.inside:
        return True

    # Inside, rear-access, or flat — need floor or lift situation
    if access.inside or access.collection_side in ("inside", "rear") or access.alley_inside:
        if access.has_floor:
            return True
        if access.has_lift_info:
            return True
        return False

    # Bungalow mentioned
    if access.floor_label == "bungalow":
        return True

    return False


def _location_acknowledgement(access: CollectionAccess) -> str:
    if access.customer_phrase:
        phrase = _clean_location_phrase(access.customer_phrase.strip())
        if phrase and not phrase[0].isupper():
            phrase = phrase[0].lower() + phrase[1:]
        return f"Thank you for confirming the address — I noted that {phrase}."

    kind_ack: dict[str, str] = {
        "alley": (
            "Thank you for confirming the address — I understand the items will be "
            "collected from the alley (outside)."
        ),
        "ginnel": (
            "Thank you for confirming the address — I understand access is via the ginnel/passage "
            "at the rear."
        ),
        "snicket": "Thank you for confirming the address — I understand access is via the snicket at the rear.",
        "passage": "Thank you for confirming the address — I understand access is via a passage at the rear.",
        "rear": "Thank you for confirming the address — I understand access is round the back of the property.",
        "mews": "Thank you for confirming the address — I understand this is mews/rear-access property.",
        "garden": "Thank you for confirming the address — I understand the items are in the garden.",
        "shed": "Thank you for confirming the address — I understand the items are in the shed.",
        "garage": "Thank you for confirming the address — I understand the items are in the garage.",
        "drive": "Thank you for confirming the address — I understand the items are on the drive.",
        "kerbside": "Thank you for confirming the address — I understand the items will be left outside/kerbside.",
        "flat": "Thank you for confirming the address — I understand the items are within the flat/property.",
        "communal": (
            "Thank you for confirming the address — I understand there is communal/shared access "
            "to the property."
        ),
        "bungalow": "Thank you for confirming the address — I understand this is a bungalow (ground level).",
    }
    if access.access_kind and access.access_kind in kind_ack:
        return kind_ack[access.access_kind]
    return "Thank you for confirming the address."


def _location_follow_up_questions(access: CollectionAccess) -> str:
    questions: list[str] = []

    if access.collection_side in ("outside", "garden") and access.access_kind in (
        "garden",
        "shed",
        "garage",
        "drive",
        "kerbside",
        "alley",
    ):
        if access.access_kind == "garden" and "shed" not in (access.customer_phrase or "").lower():
            return (
                "Please could you confirm exactly where in the garden the items are, "
                "and whether our team can reach them without entering the property."
            )
        return ""

    need_floor = access.inside or access.collection_side in ("inside", "rear") or access.alley_inside
    if need_floor and not access.has_floor:
        if access.access_kind in ("ginnel", "snicket", "passage", "rear", "mews"):
            questions.append("which floor the items are on")
        elif access.access_kind == "flat":
            questions.append("which floor the flat is on")
        elif access.access_kind == "communal":
            questions.append("which floor the items are on")
        else:
            questions.append("which floor the items are on")

    if need_floor and not access.has_lift_info:
        questions.append("whether there is a lift available (or if it is stairs only)")

    if not questions:
        return (
            "Please let me know if you need the items collected from outside or inside "
            "(ground floor, 1st floor, etc.) and if there is a lift available."
        )

    if len(questions) == 1:
        return f"Please could you confirm {questions[0]}?"
    return f"Please could you confirm {questions[0]} and {questions[1]}?"


def _variant_location_contextual(
    name: str,
    access: CollectionAccess,
    *,
    first_contact: bool = True,
    ask_dismantling: bool = False,
) -> str:
    ack = _location_acknowledgement(access)
    follow_up = _location_follow_up_questions(access)
    body = ack if follow_up else ack
    if follow_up:
        body = f"{ack}\n\n{follow_up}"
    if ask_dismantling:
        body = (
            f"{body}\n\n"
            "Also, please let me know if any of the items will need to be dismantled "
            "before we can remove them."
        )
    return (
        f"Hi {name},\n\n"
        f"{_reply_opener(first_contact=first_contact)}\n\n"
        f"{body}\n\n"
        "Kind regards."
    )


def _has_collection_location(text: str, address: str = "") -> bool:
    return _collection_location_complete(_parse_collection_access(text, address))


def _mentions_dismantle(text: str) -> bool:
    return "dismantl" in (text or "").lower()


def _items_suggest_dismantle(comments: str) -> bool:
    if _mentions_dismantle(comments):
        return False
    low = (comments or "").lower()
    return any(
        w in low
        for w in (
            "bed frame",
            "bed ",
            "wardrobe",
            "sofa bed",
            "kitchen unit",
            "cabinet",
            "cupboard",
            "desk",
            "table",
            "shed",
        )
    )


def _access_hint_from_comments(comments: str, address: str = "") -> str | None:
    access = _parse_collection_access(comments, address)
    if access.customer_phrase:
        return access.customer_phrase.lower()
    hints = {
        "ginnel": "the ginnel/rear passage",
        "snicket": "the snicket at the rear",
        "alley": "the alley (outside collection)",
        "passage": "the passage at the rear",
        "rear": "rear access round the back",
        "mews": "mews/rear access",
        "garden": "garden or shed access",
        "shed": "the garden shed",
        "garage": "the garage",
        "drive": "the drive",
        "communal": "communal/shared access",
        "flat": "the flat",
    }
    if access.access_kind in hints:
        return hints[access.access_kind]
    low = (comments or "").lower()
    if "loft" in low and "ladder" in low:
        return "loft access (ladder required)"
    if "loft" in low:
        return "loft access"
    return None


_LOCATION_SCHEDULING_ONLY_RE = re.compile(
    r"\b(?:"
    r"ground level|front door|just inside|from inside|no need for a lift|no lift|"
    r"price is fine|when works|available this afternoon|this afternoon or"
    r")\b",
    re.I,
)


def _is_location_or_scheduling_only(text: str) -> bool:
    """Reply is about access, lift, or booking — not what needs collecting."""
    snippet = _clean_customer_item_snippet(text or "")
    if not snippet:
        return False
    low = snippet.lower()
    if not _LOCATION_SCHEDULING_ONLY_RE.search(low):
        return False
    return not any(
        w in low
        for w in (
            "sofa",
            "mattress",
            "fridge",
            "bag",
            "rubbish",
            "wardrobe",
            "table",
            "chair",
            "desk",
            "bed",
            "tv",
            "washer",
            "cooker",
            "box",
            "tile",
            "beam",
            "cardboard",
        )
    )


def _job_summary(comments: str) -> str:
    text = re.sub(r"\s+", " ", _clean_customer_item_snippet(comments or ""))
    if not text:
        return "the items described"
    if _is_location_or_scheduling_only(text):
        return "the items described"
    low = text.lower()
    m = re.search(r"\bconsisting of\s+(.+?)(?:\.|$)", text, re.I)
    if m:
        return f"the items described ({m.group(1).strip()})"
    if "bag" in low and ("rubbish" in low or "waste" in low or "mixed" in low):
        return "the bags of rubbish described"
    if "steel beam" in low or ("steel" in low and "beam" in low):
        return "the steel beam"
    if "sofa" in low and "mattress" in low:
        return "the sofa and mattress"
    if "sofa" in low:
        return "the sofa"
    if "mattress" in low:
        return "the mattress"
    if "pram" in low and "bag" in low:
        return "the bags and pram"
    if "tile" in low and "bag" in low:
        return "the bags and heavier items (such as the tiles)"
    text = re.sub(
        r"^(?:hello|hi|dear)(?:\s+team)?[^.!?]*[.!?]\s*",
        "",
        text,
        count=1,
        flags=re.I,
    )
    text = re.sub(
        r"^(?:can you give me a quote[^.!?]*[.!?]\s*)+",
        "",
        text,
        count=1,
        flags=re.I,
    )
    text = re.sub(r"^(?:its?|it is)\s+", "", text, flags=re.I)
    first = re.split(r"[.!?]\s+", text)[0].strip()
    if len(first) <= 90:
        return first.lower()
    return "the items described"


def _lift_profile(comments: str) -> dict[str, Any]:
    """
    Tailor carrier / one-hand questions to what the customer listed.
    Returns carriers_question, optional one_hand_question, item_ref.
    """
    low = (comments or "").lower()
    item_ref = _job_summary(comments)

    multi = bool(
        re.search(r"\b(\d+\s+bags?|several|multiple|mixed|and|plus|,)\b", low)
        or (low.count(" and ") >= 2)
    )
    heavy = any(
        w in low
        for w in (
            "steel beam",
            "beam",
            "piano",
            "safe",
            "hot tub",
            "aga ",
            "concrete",
            "rubble",
            "bath",
            "fridge",
            "american fridge",
            "wardrobe",
            "sofa",
            "tiles",
            "heavy",
        )
    )
    light = any(w in low for w in ("bag", "rubbish", "cardboard", "boxes", "small"))

    if multi and heavy and light:
        carriers = (
            "Can one person carry the lighter items, or will the heavier pieces "
            "(such as the tiles or bulky items) require 2 people — or more?"
        )
        one_hand = None
    elif "bag" in low and heavy:
        carriers = (
            "Can one person carry the bags, or will the heavier items require 2 people?"
        )
        one_hand = None
    elif "bag" in low:
        carriers = "Can one person carry the bags, or will it require 2 people?"
        one_hand = None
    elif "steel beam" in low or ("beam" in low and "steel" in low):
        carriers = "Can one person carry the steel beam, or will it require 2 people?"
        one_hand = None
    elif "piano" in low or "safe" in low:
        carriers = "Will this require 2 people to move, or more than 2 people?"
        one_hand = None
    elif "sofa" in low or "wardrobe" in low or "fridge" in low:
        carriers = f"Can one person carry {item_ref}, or will it require 2 people?"
        one_hand = None
    elif "mattress" in low:
        carriers = "Can one person carry the mattress, or will it require 2 people?"
        one_hand = None
    elif (
        ("door" in low and "front door" not in low)
        or "chair" in low
        or "desk" in low
    ):
        carriers = f"Can one person carry {item_ref}, or will it require 2 people?"
        one_hand = f"Can {item_ref} be carried using one hand?"
    elif multi:
        carriers = f"Can one person carry {item_ref}, or will it require 2 people?"
        one_hand = None
    else:
        carriers = "Can one person carry the item, or will it require 2 people?"
        one_hand = "Can the item be carried using one hand?"

    return {
        "carriers_question": carriers,
        "one_hand_question": one_hand,
        "item_ref": item_ref,
    }


def _variant_weight_and_lift(
    name: str,
    comments: str,
    *,
    include_price: bool = True,
    is_follow_up: bool = False,
    thank_for_photos: bool = False,
    ask_dismantling: bool = False,
) -> str:
    lift = _lift_profile(comments)
    lines = [f"Hi {name},", ""]

    if is_follow_up:
        lines.append("Thank you for getting back to me.")
        lines.append("")
        if thank_for_photos:
            lines.append("Thank you for the photos.")
            lines.append("")
    else:
        lines.append("I appreciate the details you have already provided.")
        lines.append("")

    if include_price:
        lines.extend(
            [
                "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.",
                "",
            ]
        )

    lines.extend(
        [
            "In order to give a completely accurate and fair quote it would help to know the",
            "approximate weight of your items, because this is a significant variable for our",
            "recycling fees.",
            "",
            "I completely understand this is not a straightforward task, and not expecting you",
            "to weigh the items, but it would be great if you could let me know the following:",
            "",
            lift["carriers_question"],
        ]
    )
    if lift.get("one_hand_question"):
        lines.append(lift["one_hand_question"])
    if ask_dismantling and not _mentions_dismantle(comments):
        lines.append(
            "Also, please let me know if any of the items will need to be dismantled "
            "before we can remove them."
        )
    lines.extend(
        [
            "",
            "If you are still unsure, it's nothing to worry about, we can amend the price when",
            "our team arrives if necessary.",
            "",
            "Kind regards.",
        ]
    )
    return "\n".join(lines)


def _variant_intake_price_and_location(
    name: str,
    comments: str,
    *,
    has_photos: bool = False,
    ask_dismantling: bool = False,
) -> str:
    """Compact intake reply — approximate price plus collection location questions."""
    thanks = "Thank you for the photos.\n\n" if has_photos else ""
    dismantle = ""
    if ask_dismantling:
        dismantle = (
            "\n\nAlso, please let me know if any of the items will need to be dismantled "
            "before we can remove them."
        )
    return (
        f"Hi {name},\n\n"
        "Thank you for getting in touch.\n\n"
        f"{thanks}"
        "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.\n\n"
        "Please let me know if you need the items collected from outside or inside "
        f"(ground floor, 1st floor, etc.) and if there is a lift available.{dismantle}\n\n"
        "Kind regards."
    )


def _reply_opener(*, first_contact: bool) -> str:
    if first_contact:
        return "Thank you for getting in touch."
    return "Thank you for getting back to me."


def _variant_location_standard(
    name: str,
    *,
    thank_for_photos: bool = False,
    first_contact: bool = True,
    ask_dismantling: bool = False,
) -> str:
    thanks = "Thank you for the photos.\n\n" if thank_for_photos else ""
    dismantle = ""
    if ask_dismantling:
        dismantle = (
            "\n\nAlso, please let me know if any of the items will need to be dismantled "
            "before we can remove them."
        )
    return (
        f"Hi {name},\n\n"
        f"{_reply_opener(first_contact=first_contact)}\n\n"
        f"{thanks}"
        "Please let me know if you need the items collected from outside or inside "
        f"(ground floor, 1st floor, etc.) and if there is a lift available.{dismantle}\n\n"
        "Kind regards."
    )


def _variant_location_access(
    name: str,
    access_hint: str,
    *,
    first_contact: bool = True,
) -> str:
    photo_line = "Thanks for the photos.\n\n" if first_contact else ""
    return (
        f"Hi {name},\n\n"
        f"{_reply_opener(first_contact=first_contact)}\n\n"
        f"{photo_line}"
        f"Please could you confirm access for {access_hint} "
        "(inside or outside, floor level, lift, and how we reach the items)?\n\n"
        "Kind regards."
    )


def _variant_location_dismantle(
    name: str,
    item_ref: str,
    *,
    first_contact: bool = True,
) -> str:
    return (
        f"Hi {name},\n\n"
        f"{_reply_opener(first_contact=first_contact)}\n\n"
        f"Regarding {item_ref}, please let me know whether collection is from outside or inside "
        "(and which floor), whether there is a lift available, and if the items are inside "
        "whether anything will need to be dismantled before we can remove them.\n\n"
        "Kind regards."
    )


def _strip_customer_reply(text: str) -> str:
    parts = re.split(
        r"\bFrom:\s*London Waste Management\b|\bFrom:\s*hello@londonwastemanagement\b|"
        r"\bon .+?wrote:|\blondon waste management support\b.*\bwrote:",
        text,
        maxsplit=1,
        flags=re.I | re.S,
    )
    return parts[0].strip()


def _clean_customer_item_snippet(text: str) -> str:
    snippet = _strip_mobile_signature(_strip_customer_reply(text or ""))
    snippet = re.split(r"\byahoo mail\b", snippet, maxsplit=1, flags=re.I)[0].strip()
    snippet = re.sub(r"\b\d+\s+photos?\s+have\s+been\s+sent\.?\s*$", "", snippet, flags=re.I).strip()
    snippet = re.sub(r"\bmany thanks\b.*$", "", snippet, flags=re.I).strip()
    return snippet.strip()


def _resolve_item_comments(
    form: dict[str, str],
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> str:
    """Consolidated item description from the full thread (LLM or content-scored rules)."""
    return resolve_thread_item_context(form, thread, content_main, from_header).description


def _is_form_body(text: str) -> bool:
    low = (text or "").lower()
    return "first name:" in low and (
        "comments:" in low or "phone number:" in low or "email:" in low
    )


def _form_body_from_context(
    thread: list[dict[str, Any]] | None,
    content_main: str,
) -> str:
    for msg in thread or []:
        body = (msg.get("contentMain") or "").strip()
        if _is_form_body(body):
            return body
    if _is_form_body(content_main):
        return content_main
    return ""


def _parse_phone_from_text(text: str) -> str | None:
    body = _strip_email_client_footer(_strip_customer_reply(text or "")).strip()
    if not body:
        return None
    m = re.search(
        r"(?:my\s+)?(?:number|phone|mobile|contact)(?:\s+is)?\s*:?\s*(0[\d\s\-().]{10,16})\b",
        body,
        re.I,
    )
    if m:
        digits = re.sub(r"\D", "", m.group(1))
        if re.fullmatch(r"0\d{10,11}", digits):
            return digits
    digits_only = re.sub(r"\D", "", body)
    if re.fullmatch(r"07\d{9}", digits_only):
        return digits_only
    if re.fullmatch(r"0\d{10,11}", digits_only):
        return digits_only
    m = re.search(r"\b(0[\d\s\-().]{10,16})\b", body)
    if m:
        digits = re.sub(r"\D", "", m.group(1))
        if re.fullmatch(r"0\d{10,11}", digits):
            return digits
    return None


def _trim_address_candidate(candidate: str) -> str:
    c = re.sub(r"\s+", " ", (candidate or "").strip(" .,"))
    return _strip_email_client_footer(c).strip(" .,")


def _has_uk_postcode(text: str) -> bool:
    candidate = (text or "").strip()
    if re.search(r"\b[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}\b", candidate, re.I):
        return True
    compact = re.sub(r"\s+", "", candidate.upper())
    return bool(re.search(r"[A-Z]{1,2}\d{1,2}[A-Z]?\d[A-Z]{2}$", compact))


def _append_postcode_from_body(candidate: str, body: str) -> str:
    """Attach a UK postcode from the message when the street match omitted it."""
    pc = re.search(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d?[A-Z]{2,3})\b", body, re.I)
    if not pc:
        return candidate
    raw = re.sub(r"\s+", "", pc.group(1).upper())
    if len(raw) < 5:
        return candidate
    compact_candidate = re.sub(r"\s+", "", candidate.upper())
    if raw in compact_candidate:
        return candidate
    if len(raw) >= 6:
        inward = raw[-3:]
        outward = raw[:-3]
        return f"{candidate} {outward} {inward}".strip()
    return f"{candidate} {raw}".strip()


def _parse_address_from_text(text: str) -> str | None:
    body = _strip_email_client_footer(_strip_customer_reply(text or "")).strip()
    if not body:
        return None
    m = re.search(
        r"(?:i'?m at|i am at|address is|located at|collect(?:ion)? from)\s+(.+?)(?:\.\s|\.?$|regards\b|thank you\b|my number\b|phone\b)",
        body,
        re.I | re.S,
    )
    if m:
        addr = _trim_address_candidate(m.group(1))
        if len(addr) >= 6 and _is_plausible_address(addr):
            return addr
    m = re.search(
        rf"\b(\d+[a-z]?\s+[\w\s'-]*{_STREET_SUFFIX}[^.\n]*)",
        body,
        re.I,
    )
    if m:
        candidate = _append_postcode_from_body(_trim_address_candidate(m.group(1)), body)
        if _is_plausible_address(candidate):
            return candidate
    m = re.search(
        r"\b([A-Za-z0-9\s'-]+?\s+[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b",
        body,
        re.I,
    )
    if m:
        candidate = _trim_address_candidate(m.group(1))
        if _is_plausible_address(candidate):
            return candidate
    m = re.search(
        rf"\b(\d+[a-z]?\s+[\w\s'-]+?\s+[A-Z]{{1,2}}\d{{1,2}}[A-Z]?\d?[A-Z]{{2,3}})\b",
        body,
        re.I,
    )
    if m:
        candidate = _append_postcode_from_body(_trim_address_candidate(m.group(1)), body)
        if _is_plausible_address(candidate):
            return candidate
    return None


def _is_plausible_address(addr: str) -> bool:
    candidate = (addr or "").strip()
    if len(candidate) < 8:
        return False
    low = candidate.lower()
    if any(x in low for x in ("yahoo mail", "mail:", "organise", "conquer", "gmail")):
        return False
    if "?" in candidate:
        return False
    has_number = bool(re.search(r"\b\d+[a-z]?\b", candidate))
    return has_number or _has_uk_postcode(candidate)


def _enrich_form_from_customer(
    form: dict[str, str],
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> dict[str, str]:
    bodies = _customer_reply_bodies(thread, content_main, from_header)
    combined = "\n".join(bodies)

    if not form.get("address"):
        best_addr: str | None = None
        for body in bodies:
            addr = _parse_address_from_text(body)
            if not addr or not _is_full_collection_address(addr):
                continue
            if not best_addr or len(addr) > len(best_addr):
                best_addr = addr
        if best_addr:
            form["address"] = best_addr

    if not form.get("phone"):
        for body in reversed(bodies):
            phone = _parse_phone_from_text(body)
            if phone:
                form["phone"] = phone
                break

    if not form.get("email"):
        email = parse_email_address(from_header)
        if email and not is_lwm_address(from_header):
            form["email"] = email

    display = parse_display_name(from_header)
    if display:
        parts = display.split()
        if parts and not form.get("first_name"):
            form["first_name"] = parts[0]
        if len(parts) > 1 and not form.get("last_name"):
            form["last_name"] = " ".join(parts[1:])

    return form


def _merge_form(
    parsed_form: dict[str, str] | None,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str = "",
) -> dict[str, str]:
    merged: dict[str, str] = dict(parsed_form or {})
    form_body = _form_body_from_context(thread, content_main)
    for source in (form_body, content_main):
        if not source:
            continue
        for key, value in parse_form_fields(source).items():
            merged.setdefault(key, value)
    return _enrich_form_from_customer(merged, thread, content_main, from_header)


def _customer_reply_bodies(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> list[str]:
    bodies: list[str] = []
    seen: set[str] = set()
    rows = list(thread or [])
    if content_main and not _is_form_body(content_main):
        rows.append({"contentMain": content_main, "from": from_header})
    for msg in rows:
        body = _strip_customer_reply((msg.get("contentMain") or "").strip())
        from_h = msg.get("from") or msg.get("from_address") or ""
        if not body or _is_form_body(body) or is_lwm_address(from_h):
            continue
        key = body[:240].lower()
        if key in seen:
            continue
        seen.add(key)
        bodies.append(body)
    return bodies


def _latest_customer_message(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> str:
    bodies = _customer_reply_bodies(thread, content_main, from_header)
    return bodies[-1] if bodies else (content_main or "")


def _customer_providing_contact_only(text: str) -> bool:
    """Latest message is mainly address and/or phone — not a new scheduling question."""
    body = _strip_email_client_footer(_strip_customer_reply(text or "")).strip()
    if not body:
        return False
    low = body.lower()
    if any(
        w in low
        for w in (
            "when",
            "earliest",
            "availability",
            "saturday",
            "sunday",
            "tomorrow",
            "today",
            "time slot",
            "what time",
            "proceed",
            "arrange",
        )
    ):
        return False
    has_phone = bool(_parse_phone_from_text(body))
    has_address = bool(_parse_address_from_text(body))
    if not has_phone and not has_address:
        return False
    return len(low) <= 90


def _scheduling_intent_text(
    latest: str,
    thread_customer_text: str,
) -> str:
    """Use the latest customer turn for scheduling; avoid stale thread-wide triggers."""
    if _customer_providing_contact_only(latest):
        return latest
    if (
        customer_asks_availability(latest)
        or customer_asks_slot_options(latest)
        or _customer_wants_to_proceed(latest)
    ):
        return latest
    return thread_customer_text


def _is_lwm_outbound_message(msg: dict[str, Any]) -> bool:
    body = (msg.get("contentMain") or "").strip()
    if not body or _is_form_body(body):
        return False
    direction = (msg.get("direction") or "").strip().lower()
    if direction == "outbound":
        return True
    from_h = msg.get("from") or msg.get("from_address") or ""
    return is_lwm_address(from_h)


def _lwm_reply_bodies(thread: list[dict[str, Any]] | None) -> list[str]:
    bodies: list[str] = []
    for msg in thread or []:
        body = (msg.get("contentMain") or "").strip()
        if not body or not _is_lwm_outbound_message(msg):
            continue
        bodies.append(body)
    return bodies


def _lwm_asked_photos(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies).lower()
    return any(
        m in combined
        for m in (
            "relevant photos",
            "provide photos",
            "provide all relevant photos",
            "send photos",
            "send pictures",
            "upload",
            "clear photos",
        )
    )


def _lwm_phone_quote_context(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies).lower()
    return any(
        m in combined
        for m in (
            "speaking to you over the phone",
            "spoke on the phone",
            "spoke to you on the phone",
            "on the phone",
            "phone call",
        )
    )


def _is_picture_for_quote_thread(
    subject: str,
    thread: list[dict[str, Any]] | None,
) -> bool:
    subjects = [subject or ""]
    for msg in thread or []:
        subjects.append(msg.get("subject") or "")
    return any(
        "picture" in s.lower() and "quote" in s.lower() for s in subjects if s
    )


def _photos_sent_via_phone(
    *,
    subject: str,
    thread: list[dict[str, Any]] | None,
    lwm_bodies: list[str],
    is_phone_quote: bool,
    has_address: bool,
    has_phone: bool,
) -> bool:
    """
    Phone/WhatsApp quote: photos are often sent on the phone, not in the email thread.
    Do not re-ask for photos when the thread indicates a phone picture quote.
    """
    if not is_phone_quote:
        return False
    if _is_picture_for_quote_thread(subject, thread):
        return True
    if _lwm_phone_quote_context(lwm_bodies) and (has_address or has_phone):
        return True
    return False


def _lwm_asked_contact_details(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies).lower()
    return "full address" in combined or "contact number" in combined


def _lwm_already_quoted_price(lwm_bodies: list[str]) -> bool:
    return _lwm_quoted_price_gbp(lwm_bodies) is not None


def _lwm_quoted_price_gbp(lwm_bodies: list[str]) -> float | None:
    """Latest £ figure already quoted by LWM in the thread (+VAT or INCL VAT)."""
    combined = "\n".join(lwm_bodies)
    found: list[float] = []
    for m in re.finditer(
        r"£\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*VAT|INCL\.?\s*VAT|INCLUDING\s*VAT)",
        combined,
        re.I,
    ):
        found.append(float(m.group(1)))
    for m in re.finditer(r"£\s*(\d+(?:\.\d+)?)\s*\+?\s*VAT", combined, re.I):
        found.append(float(m.group(1)))
    for m in re.finditer(
        r"(?:initial estimation would be|we charge|charge of|estimate(?:d)?(?:\s+at)?)\s*£?\s*(\d+(?:\.\d+)?)",
        combined,
        re.I,
    ):
        found.append(float(m.group(1)))
    return found[-1] if found else None


def _lwm_price_is_incl_vat(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies)
    return bool(
        re.search(
            r"(?:£\s*)?\d+(?:\.\d+)?\s*(?:INCL\.?\s*VAT|INCLUDING\s*VAT)",
            combined,
            re.I,
        )
    )


def _price_line_for_template(lwm_bodies: list[str]) -> str:
    if _lwm_price_is_incl_vat(lwm_bodies):
        return "We charge £[PRICE] including VAT for the collection described."
    return "We charge £[PRICE]+VAT for the collection described."


def _lwm_offered_slot_from_thread(lwm_bodies: list[str]) -> str | None:
    """Latest LWM offer such as 'tomorrow between 7am-12pm'."""
    for body in reversed(lwm_bodies):
        m = re.search(
            r"(?:can send a team|send a team)\s+"
            r"(tomorrow|today|(?:on )?(?:\w+day \d{1,2} \w+))"
            r"\s+(?:between|for the anytime slot \(between)\s+([^.]+?)"
            r"(?:\.| so I can| please)",
            body,
            re.I,
        )
        if m:
            day = m.group(1).strip()
            slot = m.group(2).strip().replace("-", "–")
            return f"{day} between {slot}"
        m = re.search(
            r"(?:availability for|have availability for)\s+(today and tomorrow|tomorrow(?: and the weekend)?)",
            body,
            re.I,
        )
        if m:
            return m.group(1).strip()
    return None


def _customer_accepted_quote(text: str) -> bool:
    """Customer confirmed they are happy with the quoted price."""
    low = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:"
            r"price is fine|price(?:'s| is) ok|happy with the (?:price|quote)|"
            r"(?:that(?:'s| is)|the price is) fine|i(?:'m| am) happy with|"
            r"(?:sounds|looks) good|go ahead with (?:the )?quote|please book"
            r")\b",
            low,
        )
    )


def _quote_intake_sufficient(collected: "CollectedQuoteSlots") -> bool:
    """Photos plus item list, or a phone quote where LWM already priced from photos."""
    if not collected.has_photos:
        return False
    if collected.has_item_description:
        return True
    return bool(collected.is_phone_quote and collected.lwm_already_quoted_price)


def _customer_wants_to_proceed(text: str) -> bool:
    """Customer is ready to book after receiving a quote."""
    low = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:"
            r"like to proceed|would like to proceed|happy to proceed|please proceed|"
            r"please book|book(?:ing)? in|go ahead|when(?:'s| is) the earliest|"
            r"what(?:'s| is) the earliest|when can you do|how soon|asap|as soon as"
            r")\b",
            low,
        )
    )


def _is_light_collection_items(comments: str) -> bool:
    """Small/light loads where weight/lift questions are usually unnecessary."""
    low = (comments or "").lower()
    if not low:
        return False
    light = (
        "cardboard",
        "box",
        "boxes",
        "bag",
        "bags",
        "rubbish",
        "paper",
        "packaging",
        "small",
    )
    heavy = (
        "sofa",
        "mattress",
        "fridge",
        "freezer",
        "wardrobe",
        "piano",
        "beam",
        "safe",
        "bath",
        "cooker",
        "american fridge",
        "hot tub",
    )
    if not any(w in low for w in light):
        return False
    return not any(w in low for w in heavy)


def _weight_lift_waived(
    collected: "CollectedQuoteSlots",
    customer_text: str,
    *,
    comments: str = "",
    thread_customer_text: str = "",
) -> bool:
    """Skip re-asking weight/lift when the thread already supports booking."""
    context = thread_customer_text or customer_text
    if _customer_accepted_quote(context) or _customer_wants_to_proceed(context):
        return True
    if customer_asks_availability(context) and collected.lwm_already_quoted_price:
        return True
    if collected.lwm_already_quoted_price and _is_light_collection_items(comments):
        return True
    return False


def _collection_location_waived(
    collected: "CollectedQuoteSlots",
    *,
    comments: str = "",
) -> bool:
    """Standard rules for skipping inside/outside when booking can proceed."""
    if collected.has_collection_location:
        return True
    if not (
        collected.has_address
        and collected.has_phone
        and collected.lwm_already_quoted_price
        and _quote_intake_sufficient(collected)
    ):
        return False
    if _is_light_collection_items(comments):
        return True
    access = collected.access
    if access and access.collection_side in ("outside", "garden", "kerbside", "drive"):
        return True
    return False


@dataclass
class BookingPipelineContext:
    """Thread-wide context used for every booking decision."""

    thread_customer_text: str
    latest_customer_message: str
    comments: str
    lwm_bodies: list[str]


@dataclass
class BookingGapAnalysis:
    missing: list[str]
    ready_for_availability: bool
    ready_for_confirmation: bool


def _build_booking_pipeline_context(
    *,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
    comments: str = "",
) -> BookingPipelineContext:
    return BookingPipelineContext(
        thread_customer_text="\n".join(
            _customer_reply_bodies(thread, content_main, from_header)
        ),
        latest_customer_message=_latest_customer_message(
            thread, content_main, from_header
        ),
        comments=comments,
        lwm_bodies=_lwm_reply_bodies(thread),
    )


def _thread_has_scheduling_context(ctx: BookingPipelineContext) -> bool:
    """Thread is in the booking/scheduling phase (not just a fresh quote)."""
    text = ctx.thread_customer_text
    latest = ctx.latest_customer_message
    if customer_asks_availability(text) or customer_asks_availability(latest):
        return True
    if customer_states_preferred_date_or_slot(text) or customer_states_preferred_date_or_slot(
        latest
    ):
        return True
    if _customer_wants_to_proceed(text) or _customer_accepted_quote(text):
        return True
    lwm = "\n".join(ctx.lwm_bodies).lower()
    if any(
        marker in lwm
        for marker in (
            "we can send a team",
            "time slot",
            "availability for",
            "between 7am",
            "between 12pm",
            "morning slot",
            "afternoon slot",
            "fully booked for",
            "book you in",
            "payment link",
            "invoice with the payment link",
        )
    ):
        return True
    if _lwm_asked_contact_details(ctx.lwm_bodies):
        latest = ctx.latest_customer_message
        if _customer_providing_contact_only(latest):
            return True
        if _parse_phone_from_text(latest) or _parse_address_from_text(latest):
            return True
    return False


def _weight_requirements_met(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> bool:
    if _weight_lift_waived(
        collected,
        ctx.latest_customer_message,
        comments=ctx.comments,
        thread_customer_text=ctx.thread_customer_text,
    ):
        return True
    weight_ok = (
        collected.weight_kg is not None
        or collected.weight_vague
        or collected.has_item_weights
    )
    lift_ok = collected.lift_answer is not None
    access = collected.access
    if not lift_ok and access and access.has_lift_info:
        lift_ok = True
    return weight_ok and lift_ok


def _location_requirements_met(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> bool:
    if collected.has_collection_location:
        return True
    return _collection_location_waived(collected, comments=ctx.comments)


def _contact_requirements_met(collected: CollectedQuoteSlots) -> bool:
    return collected.has_address and collected.has_phone


def _analyze_booking_gaps(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> BookingGapAnalysis:
    """
    Standard booking pipeline for all quote threads.

    Stages (in order): intake → contact → location → weight/lift → availability → confirmation.
    """
    text = ctx.thread_customer_text
    missing: list[str] = []

    if not collected.has_photos:
        missing.append("item_photos")
    if not collected.has_item_description:
        missing.append("item_list_in_comments")
    if not collected.has_address:
        missing.append("address")
    if not collected.has_phone:
        missing.append("phone")
    if not _location_requirements_met(collected, ctx):
        missing.append("collection_location")

    needs_weight_lift = (
        collected.has_item_description
        or collected.lwm_asked_weight_lift
        or collected.is_phone_quote
        or (collected.lwm_asked_photos and (collected.has_address or collected.has_phone))
    )
    if needs_weight_lift and not _weight_requirements_met(collected, ctx):
        if collected.lift_answer is None:
            missing.append("lift_requirement")
        if (
            collected.weight_kg is None
            and not collected.weight_vague
            and not collected.has_item_weights
        ):
            missing.append("item_weight")

    intake_ok = _quote_intake_sufficient(collected)
    contact_ok = _contact_requirements_met(collected)
    location_ok = _location_requirements_met(collected, ctx)
    weight_ok = _weight_requirements_met(collected, ctx)
    price_ok = collected.lwm_already_quoted_price
    scheduling_ok = _thread_has_scheduling_context(ctx)

    ready_for_availability = (
        intake_ok
        and contact_ok
        and location_ok
        and weight_ok
        and price_ok
        and scheduling_ok
    )

    customer_wants_slots = (
        customer_asks_availability(text)
        or customer_asks_availability(ctx.latest_customer_message)
        or customer_states_preferred_date_or_slot(text)
        or customer_states_preferred_date_or_slot(ctx.latest_customer_message)
        or (
            (_customer_accepted_quote(text) or _customer_wants_to_proceed(text))
            and price_ok
        )
    )
    latest_asks_scheduling = (
        customer_asks_availability(ctx.latest_customer_message)
        or customer_asks_slot_options(ctx.latest_customer_message)
    )
    if intake_ok and price_ok and (
        latest_asks_scheduling
        or (customer_wants_slots and not ready_for_availability)
    ):
        if not customer_confirmed_booking_slot(ctx.latest_customer_message):
            if "booking_availability" not in missing:
                missing.append("booking_availability")

    ready_for_confirmation = (
        intake_ok
        and contact_ok
        and location_ok
        and weight_ok
        and price_ok
        and scheduling_ok
        and customer_confirmed_booking_slot(ctx.latest_customer_message)
    )

    if ready_for_confirmation:
        return BookingGapAnalysis(
            missing=[],
            ready_for_availability=True,
            ready_for_confirmation=True,
        )

    return BookingGapAnalysis(
        missing=missing,
        ready_for_availability=ready_for_availability,
        ready_for_confirmation=False,
    )


def _lwm_asked_weight_lift(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies).lower()
    return any(
        m in combined
        for m in (
            "approximate weight",
            "one person carry",
            "require 2 people",
            "carried using one hand",
            "recycling fees",
            "heavy items",
            "specifically heavy",
        )
    )


def _lwm_asked_location(lwm_bodies: list[str]) -> bool:
    combined = "\n".join(lwm_bodies).lower()
    return any(
        m in combined
        for m in (
            "outside or inside",
            "ground floor",
            "lift available",
            "which floor",
        )
    )


def _parse_lift_answer(text: str) -> str | None:
    low = (text or "").lower()
    if re.search(r"\b(two people|2 people|2 persons|needs? 2|need 2|require 2)\b", low):
        return "two_people"
    if re.search(
        r"\b(one person|1 person|single person|one man|with one man|do the job with one man)\b",
        low,
    ):
        return "one_person"
    if re.search(r"\b(?:liftable|can be lifted|carried)\s+by\s+(?:an?\s+)?individual\b", low):
        return "one_person"
    if re.search(r"\blifted (?:them |everything )?myself\b", low):
        return "one_person"
    if re.search(r"\b(two hands|2 hands|both arms|two arms)\b", low):
        return "two_hands"
    if re.search(r"\b(not under arm|with two arms)\b", low):
        return "two_hands"
    return None


def _parse_vague_weight_answer(text: str) -> bool:
    """Customer acknowledged weight without giving kg — sufficient for booking."""
    low = (text or "").lower()
    if re.search(
        r"\b(?:hard|difficult)\s+to\s+say\b.{0,40}\bweight\b",
        low,
    ):
        return True
    if re.search(r"\b(?:not sure|unsure|don'?t know|no idea)\b.{0,40}\bweight\b", low):
        return True
    if re.search(r"\b(?:lifted|carried)\s+(?:them|it|everything|all)\s+myself\b", low):
        return True
    return False


def _parse_weight_kg(text: str) -> float | None:
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?|kilogram)\b", text or "", re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(?:about|around|approx(?:imately)?|maybe)\s*(\d+(?:\.\d+)?)\b", text or "", re.I)
    if m and "kg" not in (text or "").lower():
        return float(m.group(1))
    return None


def _extract_item_weights_kg(text: str) -> list[float]:
    """Per-item weights from quotation forms, e.g. 'around 20kg' and '4-5kg'."""
    if not text:
        return []
    weights: list[float] = []
    for m in re.finditer(
        r"\b(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?)\b",
        text,
        re.I,
    ):
        weights.append(float(m.group(2)))
    for m in re.finditer(
        r"\b(?:around|about|approx(?:imately)?|~)?\s*(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?)\b",
        text,
        re.I,
    ):
        if m.start() > 0 and text[m.start() - 1] == "-":
            continue
        weights.append(float(m.group(1)))
    return weights


def _infer_carrier_lift(weights: list[float]) -> str | None:
    if not weights:
        return None
    peak = max(weights)
    total = sum(weights)
    if peak >= 35 or total >= 70:
        return "two_people"
    if peak >= 18 or total >= 35:
        return "two_hands"
    return "one_person"


def _is_quote_follow_up(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> bool:
    if _customer_reply_bodies(thread, content_main, from_header):
        return True
    if thread and len(thread) > 1:
        return True
    return False


def _is_first_customer_inquiry(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
) -> bool:
    """Customer's opening message — no LWM reply in the thread yet."""
    return not _lwm_reply_bodies(thread) and bool(
        _customer_reply_bodies(thread, content_main, from_header)
    )


@dataclass
class CollectedQuoteSlots:
    has_photos: bool = False
    has_item_description: bool = False
    has_address: bool = False
    has_phone: bool = False
    lift_answer: str | None = None
    weight_kg: float | None = None
    has_collection_location: bool = False
    alley_inside: bool = False
    access: CollectionAccess | None = None
    lwm_asked_weight_lift: bool = False
    lwm_asked_location: bool = False
    lwm_asked_photos: bool = False
    lwm_asked_contact: bool = False
    is_phone_quote: bool = False
    photos_via_phone: bool = False
    lwm_already_quoted_price: bool = False
    item_description_source: str = "none"
    weight_vague: bool = False
    has_item_weights: bool = False


def _collect_slots(
    *,
    thread: list[dict[str, Any]] | None,
    content_main: str,
    from_header: str,
    form: dict[str, str],
    attachments: list[dict[str, Any]] | None,
    subject: str = "",
) -> CollectedQuoteSlots:
    form_body = _form_body_from_context(thread, content_main)
    customer_bodies = _customer_reply_bodies(thread, content_main, from_header)
    customer_text = "\n".join(customer_bodies)
    lwm_bodies = _lwm_reply_bodies(thread)

    has_photos = bool(attachments) or any(
        _customer_sent_photos(body, attachments if body == customer_bodies[-1] else None)
        for body in customer_bodies
    )
    if not has_photos and form_body:
        has_photos = _customer_sent_photos("", attachments) or _has_customer_photos(form_body, attachments)

    item_ctx = resolve_thread_item_context(form, thread, content_main, from_header)
    has_description = item_ctx.has_description

    lift_answer = None
    weight_kg = None
    weight_vague = False
    item_weights: list[float] = []
    weight_sources: list[str] = []
    comments_from_form = _comments_text(form_body) if form_body else ""
    if comments_from_form:
        weight_sources.append(comments_from_form)
    elif form.get("comments"):
        weight_sources.append(form.get("comments") or "")
    weight_sources.extend(customer_bodies)
    seen_weight_text: set[str] = set()
    for body in weight_sources:
        key = (body or "").strip()[:400]
        if not key or key in seen_weight_text:
            continue
        seen_weight_text.add(key)
        lift_answer = _parse_lift_answer(body) or lift_answer
        weight_vague = weight_vague or _parse_vague_weight_answer(body)
        item_weights.extend(_extract_item_weights_kg(body))
        if not item_weights:
            weight_kg = weight_kg or _parse_weight_kg(body)

    has_item_weights = bool(item_weights)
    if item_weights:
        weight_kg = sum(item_weights)
    if lift_answer is None and item_weights:
        lift_answer = _infer_carrier_lift(item_weights)

    address = (form.get("address") or field(form_body or "", "Address") or "").strip()
    if not address:
        for body in reversed(customer_bodies):
            addr = _parse_address_from_text(body)
            if addr:
                address = addr
                break
    comments = (form.get("comments") or _comments_text(form_body) or "").strip()
    combined_access_text = "\n".join(p for p in (comments, customer_text) if p)
    access = _parse_collection_access(combined_access_text, address)
    has_location = _collection_location_complete(access)

    phone_quote = (
        bool(lwm_bodies)
        and not form_body
        and (_lwm_asked_photos(lwm_bodies) or _lwm_phone_quote_context(lwm_bodies))
    )

    phone = (form.get("phone") or "").strip()
    if not phone:
        for body in customer_bodies:
            parsed_phone = _parse_phone_from_text(body)
            if parsed_phone:
                phone = parsed_phone
                break
    has_address = _is_full_collection_address(address)
    has_phone = bool(phone)
    photos_via_phone = _photos_sent_via_phone(
        subject=subject,
        thread=thread,
        lwm_bodies=lwm_bodies,
        is_phone_quote=phone_quote,
        has_address=has_address,
        has_phone=has_phone,
    )
    if photos_via_phone:
        has_photos = True

    return CollectedQuoteSlots(
        has_photos=has_photos,
        has_item_description=has_description,
        has_address=has_address,
        has_phone=has_phone,
        lift_answer=lift_answer,
        weight_kg=weight_kg,
        has_collection_location=has_location,
        alley_inside=access.alley_inside,
        access=access,
        lwm_asked_weight_lift=_lwm_asked_weight_lift(lwm_bodies),
        lwm_asked_location=_lwm_asked_location(lwm_bodies),
        lwm_asked_photos=_lwm_asked_photos(lwm_bodies),
        lwm_asked_contact=_lwm_asked_contact_details(lwm_bodies),
        is_phone_quote=phone_quote,
        photos_via_phone=photos_via_phone,
        lwm_already_quoted_price=_lwm_already_quoted_price(lwm_bodies),
        item_description_source=item_ctx.source,
        weight_vague=weight_vague,
        has_item_weights=has_item_weights,
    )


def _should_include_availability_suggestion(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> bool:
    gap = _analyze_booking_gaps(collected, ctx)
    return gap.ready_for_availability or "booking_availability" in gap.missing


def _follow_up_missing_slots(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> list[str]:
    return _analyze_booking_gaps(collected, ctx).missing


def _is_ready_for_availability_offer(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> bool:
    return _analyze_booking_gaps(collected, ctx).ready_for_availability


def _is_ready_for_booking_confirmation(
    collected: CollectedQuoteSlots,
    ctx: BookingPipelineContext,
) -> bool:
    return _analyze_booking_gaps(collected, ctx).ready_for_confirmation


def _is_full_collection_address(address: str) -> bool:
    addr = (address or "").strip()
    if not addr:
        return False
    low = addr.lower()
    has_street = bool(
        re.search(
            rf"\b\d+[a-z]?\s+\w[\w\s'-]*{_STREET_SUFFIX}",
            low,
        )
    )
    compact = re.sub(r"\s+", "", addr.upper())
    if re.fullmatch(r"[A-Z]{1,2}\d{1,2}[A-Z]?\d[A-Z]{2}", compact):
        return False
    if not has_street and re.search(r"\bpostcode\b", low):
        return False
    return has_street or len(addr.split()) >= 5 or _has_uk_postcode(addr)


def _missing_contact_slots(form: dict[str, str]) -> list[str]:
    missing: list[str] = []
    if not _is_full_collection_address(form.get("address") or ""):
        missing.append("address")
    if not (form.get("phone") or "").strip():
        missing.append("phone")
    if not (form.get("email") or "").strip():
        missing.append("email")
    return missing


_CONTACT_ASKS: dict[str, str] = {
    "address": "your full collection address (including postcode)",
    "phone": "your contact phone number",
    "email": "your email address",
}


def _format_missing_asks(asks: list[str]) -> str:
    if not asks:
        return ""
    if len(asks) == 1:
        return f"Please could you confirm {asks[0]}?"
    if len(asks) == 2:
        return f"Please could you confirm {asks[0]} and {asks[1]}?"
    return (
        "Please could you confirm "
        + ", ".join(asks[:-1])
        + f", and {asks[-1]}?"
    )


def _variant_booking_availability(
    name: str,
    form: dict[str, str],
    collected: CollectedQuoteSlots,
    missing: list[str],
    *,
    thread: list[dict[str, Any]] | None = None,
    content_main: str = "",
    from_header: str = "",
    include_price: bool = True,
) -> tuple[str, dict[str, Any]]:
    customer_text = "\n".join(
        _customer_reply_bodies(thread, content_main, from_header)
    )
    latest = _latest_customer_message(thread, content_main, from_header)
    scheduling_text = _scheduling_intent_text(latest, customer_text)
    plan = build_booking_availability_plan(
        customer_text, scheduling_text=scheduling_text
    )
    avail_meta = availability_metadata(plan)

    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    access = collected.access or CollectionAccess()
    if collected.has_collection_location:
        if access.floor_label and access.has_lift_info and access.lift_available:
            lines.append(
                f"Thank you for confirming — collection from the {access.floor_label} "
                "with lift access."
            )
        elif access.floor_label and access.has_lift_info and access.lift_available is False:
            lines.append(
                f"Thank you for confirming — collection from the {access.floor_label} "
                "(no lift required)."
            )
        elif access.floor_label:
            lines.append(
                f"Thank you for confirming — collection from the {access.floor_label}."
            )
        elif access.access_kind == "flat":
            lines.append("Thank you for confirming the collection location.")
    elif collected.lift_answer == "one_person":
        lines.append(
            "Thank you for confirming that one person should be able to carry the items."
        )
    elif collected.lift_answer == "two_people":
        lines.append("Thank you for confirming that two people will be needed.")
    elif collected.lift_answer == "two_hands":
        lines.append(
            "Thank you for confirming — some items may need to be carried with two hands."
        )
    if collected.weight_vague and collected.weight_kg is None:
        lines.append("Thank you for the note on weight — that is helpful.")
    if len(lines) > 4:
        lines.append("")

    if include_price:
        lines.extend(
            [
                _price_line_for_template(_lwm_reply_bodies(thread)),
                "",
            ]
        )
    lines.extend(
        [
            plan.offer_line,
            "",
        ]
    )

    asks: list[str] = []
    for slot in _missing_contact_slots(form):
        asks.append(_CONTACT_ASKS[slot])
    if "item_weight" in missing:
        asks.append("the approximate weight of the items if you are able to advise")

    ask_line = _format_missing_asks(asks)
    if ask_line:
        lines.append(ask_line)

    lines.extend(["", "Kind regards."])
    return "\n".join(lines), avail_meta


def _variant_follow_up_availability(
    name: str,
    form: dict[str, str],
    collected: CollectedQuoteSlots,
    missing: list[str],
    *,
    thread: list[dict[str, Any]] | None = None,
    content_main: str = "",
    from_header: str = "",
    comments: str = "",
    include_price: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Availability + address while weight/lift may still be collected separately."""
    customer_text = "\n".join(
        _customer_reply_bodies(thread, content_main, from_header)
    )
    latest = _latest_customer_message(thread, content_main, from_header)
    scheduling_text = _scheduling_intent_text(latest, customer_text)
    plan = build_booking_availability_plan(
        customer_text, scheduling_text=scheduling_text
    )
    avail_meta = availability_metadata(plan)

    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    if customer_states_preferred_date_or_slot(scheduling_text):
        lines.append("Thank you for confirming.")
        lines.append("")

    if include_price:
        lines.extend(
            [
                _price_line_for_template(_lwm_reply_bodies(thread)),
                "",
            ]
        )

    lines.append(plan.offer_line)
    lines.append("")

    asks: list[str] = []
    for slot in _missing_contact_slots(form):
        asks.append(_CONTACT_ASKS[slot])

    ask_line = _format_missing_asks(asks)
    if ask_line:
        lines.append(ask_line)
    elif not customer_states_preferred_date_or_slot(customer_text):
        lines.append("Please let me know if you have a date in mind.")

    lines.extend(["", "Kind regards."])
    return "\n".join(lines), avail_meta


def _variant_continue_booking_flow(name: str, *, skip_price: bool = False) -> str:
    """Confirm the quote and collect details needed to complete the booking."""
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    if not skip_price:
        lines.extend(
            [
                "We charge £[PRICE]+VAT for the collection described.",
                "",
            ]
        )
    lines.extend(
        [
            "Please confirm you would like to proceed, and share your full collection address "
            "(including postcode) and contact number so I can book this in for you.",
            "",
            "Please also let me know whether the items will be left outside or inside "
            "(ground floor, 1st floor, etc.) and if there is a lift available.",
            "",
            "Kind regards.",
        ]
    )
    return "\n".join(lines)


def _variant_booking_confirmation(
    name: str,
    form: dict[str, str],
    collected: CollectedQuoteSlots,
    *,
    availability_line: str,
) -> str:
    address = (form.get("address") or "").strip()
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    if address and _is_full_collection_address(address):
        lines.append(f"Thank you for confirming the collection address — {address}.")
    access = collected.access or CollectionAccess()
    if collected.has_collection_location and access.floor_label:
        lines.append(f"Collection from the {access.floor_label}.")
    elif collected.has_collection_location and access.collection_side:
        lines.append(f"Collection from {access.collection_side} the property.")
    if len(lines) > 4:
        lines.append("")

    if availability_line:
        lines.append(availability_line)
        lines.append("")

    lines.extend(
        [
            "Your booking is confirmed.",
            "A payment link has been issued and sent to your email address.",
            "The payment needs to be made online and in advance to secure your booking.",
            "",
            "Kind regards.",
        ]
    )
    return "\n".join(lines)


def _variant_thank_you_lwm(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for using London Waste Management.\n\n"
        "Kind regards."
    )


def _variant_contact_received_booking_progress(
    name: str,
    form: dict[str, str],
    collected: CollectedQuoteSlots,
    *,
    latest_message: str,
    availability_line: str | None = None,
) -> str:
    """Customer just sent phone/address — acknowledge and move toward booking."""
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    if _parse_phone_from_text(latest_message):
        lines.append("Thank you for confirming your contact number.")
    addr = (form.get("address") or "").strip()
    if addr and _is_full_collection_address(addr):
        lines.append(f"Thank you for confirming the collection address — {addr}.")
    elif _parse_address_from_text(latest_message):
        parsed = _parse_address_from_text(latest_message)
        if parsed:
            lines.append(f"Thank you for confirming the collection address — {parsed}.")
    if len(lines) > 4:
        lines.append("")

    if availability_line:
        lines.append(availability_line)
        lines.append("")

    lines.extend(
        [
            "Please let me know whether the items will be left outside or inside "
            "(ground floor, 1st floor, etc.) and if there is a lift available, "
            "and I will book you in and send an invoice with the payment link attached.",
            "",
            "Kind regards.",
        ]
    )
    return "\n".join(lines)


def _variant_first_contact_quote(
    name: str,
    collected: CollectedQuoteSlots,
    missing: list[str],
    *,
    availability_line: str | None = None,
    form: dict[str, str] | None = None,
) -> str:
    """Opening admin reply — greet, price, earliest availability if asked, photos, contact."""
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting in touch.",
        "",
    ]
    if collected.has_item_description:
        lines.extend(
            [
                "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.",
                "",
            ]
        )
    if availability_line:
        lines.append(availability_line.strip())
        lines.append("")
    if "item_photos" in missing:
        lines.append(
            "It would be amazing if you could provide clear photos of the items so I can "
            "ensure the price given is completely fair and accurate."
        )
        lines.append("")
        lines.append(
            "I completely understand you might be busy so if it is difficult please do not worry, "
            "we can amend the price when our team arrives if necessary."
        )
    elif "item_list_in_comments" in missing:
        lines.append(
            "Could you please provide a brief list of what needs collecting "
            "so I can confirm an accurate quote?"
        )
    contact_missing = [slot for slot in ("address", "phone") if slot in missing]
    if contact_missing:
        asks = [_CONTACT_ASKS[slot] for slot in contact_missing if slot in _CONTACT_ASKS]
        contact_line = _format_missing_asks(asks)
        if contact_line:
            lines.append("")
            lines.append(contact_line)
    lines.extend(["", "Kind regards."])
    return "\n".join(lines)


def _variant_first_contact_companion(
    name: str,
    missing: list[str],
    comments: str = "",
) -> str:
    """
    Second first-contact suggestion — complementary to _variant_first_contact_quote.
    Covers access/location, weight/lift, and dismantling only (not price, dates, photos, or contact).
    """
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting in touch.",
        "",
    ]
    paragraphs: list[str] = []

    if "collection_location" in missing:
        dismantle_extra = ""
        if _items_suggest_dismantle(comments):
            dismantle_extra = (
                " Also, please let me know if any of the items will need to be dismantled "
                "before we can remove them."
            )
        paragraphs.append(
            "To help us plan the collection, please let me know whether the items are "
            "outside or inside the property (e.g. ground floor, 1st floor, etc.) and "
            f"whether there is lift access.{dismantle_extra}"
        )
    elif _items_suggest_dismantle(comments):
        paragraphs.append(
            "Please let me know if any of the items will need to be dismantled "
            "before we can remove them."
        )

    if "item_weight" in missing or "lift_requirement" in missing:
        paragraphs.append(
            "It would also help to know the approximate weight of the items and whether "
            "one person can carry them or if two people will be needed — this affects "
            "our recycling fees and crew size."
        )

    if not paragraphs:
        paragraphs.append(
            "Once I have your photos and collection address, I will send a secure payment "
            "link to confirm your booking."
        )

    lines.append("\n\n".join(paragraphs))
    lines.extend(["", "Kind regards."])
    return "\n".join(lines)


def _rules_companion_reply(
    result: dict[str, Any],
    customer_name: str,
    *,
    item_description: str = "",
) -> str | None:
    """Deterministic second suggestion that complements (does not repeat) the primary rule reply."""
    phase = str(result.get("phase") or "")
    missing = list(result.get("missing_slots") or [])
    if phase == "first_contact":
        return _variant_first_contact_companion(
            customer_name, missing, comments=item_description
        )
    return None


def _variant_follow_up_intake_gap(
    name: str,
    collected: CollectedQuoteSlots,
    *,
    need_photos: bool,
    need_items: bool,
    first_contact: bool = False,
    include_price_with_photos: bool = False,
) -> str:
    lines = [
        f"Hi {name},",
        "",
        f"{_reply_opener(first_contact=first_contact)}",
        "",
    ]
    if collected.has_photos and need_items and not need_photos:
        lines.extend(
            [
                "Thank you for the photos.",
                "",
                "Could you please provide a brief list of what needs collecting "
                "so I can confirm an accurate quote?",
                "",
                "Kind regards.",
            ]
        )
        return "\n".join(lines)

    if need_photos and include_price_with_photos:
        lines.extend(
            [
                "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.",
                "",
            ]
        )

    asks: list[str] = []
    if need_photos:
        asks.append("send clear photos of the items")
    if need_items:
        asks.append("provide a brief list of what needs collecting")
    if len(asks) == 2:
        lines.append(
            f"Could you please {' and '.join(asks)} so I can confirm an accurate quote?"
        )
    elif asks:
        lines.append(f"Could you please {asks[0]} so I can confirm an accurate quote?")
    lines.extend(["", "Kind regards."])
    return "\n".join(lines)


def _variant_follow_up_weight_lift_generic(name: str, *, skip_price: bool = False) -> str:
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
    ]
    if not skip_price:
        lines.extend(
            [
                "Based on the information I have so far, I can give an approximate price of £[PRICE]+VAT "
                "once I know what needs collecting.",
                "",
            ]
        )
    lines.extend(
        [
            "It would also help to know the approximate weight of the items and whether one person "
            "can carry them or if two people will be needed — this affects our recycling fees and "
            "crew size.",
            "",
            "Kind regards.",
        ]
    )
    return "\n".join(lines)


def _variant_weight_and_lift_followup(
    name: str,
    comments: str,
    *,
    skip_price: bool = False,
) -> str:
    lift = _lift_profile(comments)
    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
        "Could you please let me know the approximate weight of the items?",
        "",
        lift["carriers_question"],
        "",
        "Kind regards.",
    ]
    if skip_price:
        return "\n".join(lines)

    lines = [
        f"Hi {name},",
        "",
        "Thank you for getting back to me.",
        "",
        "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.",
        "",
        "Could you please let me know the approximate weight of the items?",
        "",
        lift["carriers_question"],
        "",
        "Kind regards.",
    ]
    return "\n".join(lines)


def _variant_follow_up_price_slot(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        "Based on the information I have, I can give an approximate price of £[PRICE]+VAT.\n\n"
        "Please let me know your preferred collection date and whether you would prefer a "
        "morning or afternoon slot, and I will send you the payment link to confirm.\n\n"
        "Kind regards."
    )


def _variant_follow_up_need_photos(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        "Could you please send clear photos of the items so I can confirm an accurate quote?\n\n"
        "Kind regards."
    )


def _variant_follow_up_thanks_contact_need_photos(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        "Thank you for providing your address and contact number.\n\n"
        "Could you please send clear photos of the items so I can confirm an accurate quote?\n\n"
        "Kind regards."
    )


def _variant_follow_up_weight(name: str) -> str:
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        "Could you please advise the approximate weight (e.g. 10 kg, 15 kg)?\n\n"
        "Kind regards."
    )


def _variant_follow_up_thanks_lift(name: str, lift_answer: str) -> str:
    note = {
        "one_person": "one person should be able to carry the items",
        "two_people": "two people will be needed for the collection",
        "two_hands": "two hands may be needed depending on the person carrying",
    }.get(lift_answer, "your note about how the items can be carried")
    return (
        f"Hi {name},\n\n"
        "Thank you for getting back to me.\n\n"
        f"Thank you for confirming — {note}.\n\n"
        "Kind regards."
    )


def _follow_up_variants(
    name: str,
    form: dict[str, str],
    collected: CollectedQuoteSlots,
    missing: list[str],
    *,
    thread: list[dict[str, Any]] | None = None,
    content_main: str = "",
    from_header: str = "",
    pipeline_ctx: BookingPipelineContext | None = None,
    latest_message: str = "",
) -> tuple[list[str], dict[str, Any]]:
    comments = _resolve_item_comments(form, thread, content_main, from_header)
    replies: list[str] = []
    availability_info: dict[str, Any] = {}
    ctx = pipeline_ctx or _build_booking_pipeline_context(
        thread=thread,
        content_main=content_main,
        from_header=from_header,
        comments=comments,
    )
    latest = latest_message or ctx.latest_customer_message
    availability_text = ctx.thread_customer_text
    first_contact = _is_first_customer_inquiry(thread, content_main, from_header)

    if first_contact and (
        "item_photos" in missing or "item_list_in_comments" in missing
    ):
        availability_line: str | None = None
        if customer_asks_availability(latest) or customer_asks_availability(
            availability_text
        ):
            plan = build_booking_availability_plan(
                availability_text,
                scheduling_text=_scheduling_intent_text(latest, availability_text),
            )
            availability_info = availability_metadata(plan)
            availability_line = plan.offer_line
        replies.append(
            _variant_first_contact_quote(
                name,
                collected,
                missing,
                availability_line=availability_line,
                form=form,
            )
        )
        return cap_suggestions(replies), availability_info

    if not missing and _is_ready_for_booking_confirmation(collected, ctx):
        plan = build_booking_availability_plan(
            availability_text,
            scheduling_text=_scheduling_intent_text(latest, availability_text),
        )
        availability_info = availability_metadata(plan)
        replies.append(
            _variant_booking_confirmation(
                name,
                form,
                collected,
                availability_line=confirmation_offer_line(plan),
            )
        )
        replies.append(_variant_thank_you_lwm(name))
        return cap_suggestions(replies), availability_info

    ready_for_booking = _is_ready_for_availability_offer(collected, ctx)
    include_availability = "booking_availability" in missing
    contact_only = _customer_providing_contact_only(latest)

    need_photos = "item_photos" in missing
    need_items = "item_list_in_comments" in missing
    first_contact = _is_first_customer_inquiry(thread, content_main, from_header)
    if need_photos or need_items:
        replies.append(
            _variant_follow_up_intake_gap(
                name,
                collected,
                need_photos=need_photos,
                need_items=need_items,
                first_contact=first_contact,
                include_price_with_photos=need_photos and collected.has_item_description,
            )
        )

    if "item_weight" in missing or "lift_requirement" in missing:
        skip_price = collected.lwm_already_quoted_price
        if comments:
            replies.append(
                _variant_weight_and_lift_followup(
                    name,
                    comments,
                    skip_price=skip_price,
                )
            )
        else:
            replies.append(
                _variant_follow_up_weight_lift_generic(name, skip_price=skip_price)
            )

    if include_availability:
        variant_fn = (
            _variant_booking_availability
            if ready_for_booking
            else _variant_follow_up_availability
        )
        common_kwargs: dict[str, Any] = {
            "name": name,
            "form": form,
            "collected": collected,
            "missing": missing,
            "thread": thread,
            "content_main": content_main,
            "from_header": from_header,
        }
        if not ready_for_booking:
            common_kwargs["comments"] = comments

        avail_with_price, availability_info = variant_fn(
            **common_kwargs,
            include_price=True,
        )
        replies.append(avail_with_price)

        if not ready_for_booking and (
            "collection_location" in missing or _missing_contact_slots(form)
        ):
            replies.append(_variant_continue_booking_flow(name, skip_price=True))

    elif (
        contact_only
        and collected.has_address
        and collected.has_phone
        and collected.lwm_already_quoted_price
        and "collection_location" in missing
    ):
        contact_plan = build_booking_availability_plan(
            availability_text,
            scheduling_text=_scheduling_intent_text(latest, availability_text),
        )
        if not availability_info:
            availability_info = availability_metadata(contact_plan)
        replies.append(
            _variant_contact_received_booking_progress(
                name,
                form,
                collected,
                latest_message=latest,
                availability_line=confirmation_offer_line(contact_plan),
            )
        )

    if "collection_location" in missing:
        access = collected.access or CollectionAccess()
        ask_dismantling = not _mentions_dismantle(comments)
        if access.access_kind or access.customer_phrase or access.collection_side:
            replies.append(
                _variant_location_contextual(
                    name,
                    access,
                    first_contact=False,
                    ask_dismantling=ask_dismantling,
                )
            )
        elif _items_suggest_dismantle(comments):
            replies.append(
                _variant_location_dismantle(
                    name, _job_summary(comments), first_contact=False
                )
            )
        else:
            replies.append(
                _variant_location_standard(
                    name,
                    first_contact=False,
                    ask_dismantling=ask_dismantling,
                )
            )

    if "booking_slot" in missing:
        replies.append(_variant_follow_up_price_slot(name))

    if not replies and (
        customer_states_preferred_date_or_slot(latest)
        or customer_asks_availability(latest)
    ):
        plan = build_booking_availability_plan(
            availability_text,
            scheduling_text=_scheduling_intent_text(latest, availability_text),
        )
        availability_info = availability_metadata(plan)
        avail_reply, _ = _variant_follow_up_availability(
            name,
            form,
            collected,
            missing,
            thread=thread,
            content_main=content_main,
            from_header=from_header,
            comments=comments,
            include_price=not collected.lwm_already_quoted_price,
        )
        replies.append(avail_reply)

    if not replies:
        replies.append(_variant_follow_up_price_slot(name))

    return cap_suggestions(replies), availability_info


def _pricing_source_text(
    thread: list[dict[str, Any]] | None,
    content_main: str,
    form: dict[str, str],
    from_header: str = "",
) -> str:
    resolved = _resolve_item_comments(form, thread, content_main, from_header)
    if resolved:
        return resolved
    form_body = _form_body_from_context(thread, content_main)
    if form_body:
        return form_body
    return content_main


def _intake_complete_variants(
    name: str,
    content_main: str,
    form: dict[str, str],
    *,
    has_photos: bool,
) -> tuple[list[str], list[str]]:
    """Weight/lift reply plus optional location variants; returns (replies, missing_slots)."""
    comments = (form.get("comments") or _comments_text(content_main) or "").strip()
    address = (form.get("address") or field(content_main, "Address") or "").strip()
    analysis = _analysis_text(content_main, form)

    ask_dismantling = _items_suggest_dismantle(comments)
    replies = [
        _variant_weight_and_lift(name, comments),
        _variant_intake_price_and_location(
            name,
            comments,
            has_photos=has_photos,
            ask_dismantling=ask_dismantling,
        ),
    ]
    missing = ["item_weight", "lift_requirement"]

    if not _has_collection_location(analysis, address):
        missing.append("collection_location")
        access = _parse_collection_access(analysis, address)
        if access.access_kind or access.customer_phrase or access.collection_side:
            replies.append(_variant_location_contextual(name, access))
        elif ask_dismantling:
            replies.append(_variant_location_dismantle(name, _job_summary(comments)))
        else:
            replies.append(_variant_location_standard(name, thank_for_photos=has_photos))

    return cap_suggestions(replies), missing


def _dedupe_inserted_pricing(text: str) -> str:
    """Keep a single price mention when template and apply_pricing both reference the total."""
    lines = text.split("\n")
    price_line_re = re.compile(
        r"£\s*[\d.]+\+VAT|\bwe (estimate|charge)\b.*£|\bapproximate price of £",
        re.I,
    )
    seen = False
    kept: list[str] = []
    for ln in lines:
        if price_line_re.search(ln.strip()):
            if seen:
                continue
            seen = True
        kept.append(ln)
    return "\n".join(kept).strip()


def _pricing_access_context(
    form: dict[str, str],
    content_main: str,
    thread: list[dict[str, Any]] | None,
    *,
    collected: CollectedQuoteSlots | None = None,
    from_header: str = "",
) -> tuple[str | None, bool]:
    form_body = _form_body_from_context(thread, content_main)
    comments = (form.get("comments") or _comments_text(form_body or content_main) or "").strip()
    customer_text = "\n".join(_customer_reply_bodies(thread, content_main, from_header))
    combined = f"{comments}\n{customer_text}"

    if collected and collected.access:
        side = collected.access.collection_side
    else:
        address = (form.get("address") or field(content_main, "Address") or "").strip()
        side = _parse_collection_access(_analysis_text(content_main, form), address).collection_side

    needs_dismantling = _mentions_dismantle(combined) or _items_suggest_dismantle(comments)
    return side, needs_dismantling


_PRICE_PLACEHOLDER_RE = re.compile(r"£\s*\[PRICE\]\+VAT", re.I)
_PRICE_INCL_PLACEHOLDER_RE = re.compile(r"£\s*\[PRICE\]\s*including\s+VAT", re.I)


def _substitute_reply_prices(replies: list[str], pricing: dict[str, Any]) -> list[str]:
    from services.standard_items import _format_gbp

    ballpark = pricing.get("ballpark_gbp")
    if ballpark is None:
        return replies
    total = _format_gbp(ballpark)
    incl = bool(pricing.get("price_incl_vat"))
    updated: list[str] = []
    for reply in replies:
        text = reply or ""
        if _PRICE_INCL_PLACEHOLDER_RE.search(text):
            text = _PRICE_INCL_PLACEHOLDER_RE.sub(f"£{total} including VAT", text)
        elif _PRICE_PLACEHOLDER_RE.search(text):
            if incl:
                text = _PRICE_PLACEHOLDER_RE.sub(f"£{total} including VAT", text)
            else:
                text = apply_pricing_to_suggested_replies([text], pricing)[0]
        elif incl:
            text = re.sub(
                rf"£\s*{re.escape(str(ballpark))}\s*\+?\s*VAT",
                f"£{total} including VAT",
                text,
                flags=re.I,
            )
            text = re.sub(
                rf"£\s*{re.escape(total)}\s*\+?\s*VAT",
                f"£{total} including VAT",
                text,
                flags=re.I,
            )
        updated.append(_dedupe_inserted_pricing(text))
    return updated


def _pick_pricing_reply_index(replies: list[str]) -> int | None:
    """Apply ballpark price to weight/lift or booking templates — not intake-gap asks."""
    for i, reply in enumerate(replies):
        if _PRICE_PLACEHOLDER_RE.search(reply or ""):
            return i
    for i, reply in enumerate(replies):
        low = (reply or "").lower()
        if "we charge £" in low and "[price]" in low:
            return i
        if "approximate weight of the items" in low or "once i know what needs collecting" in low:
            return i
    return None


def _format_thread_summary_for_llm(
    thread: list[dict[str, Any]] | None,
    content_main: str,
) -> str:
    lines: list[str] = []
    for msg in thread or []:
        direction = (msg.get("direction") or "unknown").strip()
        body = (msg.get("contentMain") or "").strip()
        if body:
            lines.append(f"[{direction}] {body[:2500]}")
    if content_main and not any(content_main in ln for ln in lines):
        lines.append(f"[current] {content_main[:2500]}")
    return "\n\n".join(lines)


def _pricing_summary_for_llm(pricing: dict[str, Any] | None) -> str:
    if not pricing:
        return ""
    parts: list[str] = []
    ballpark = pricing.get("ballpark_gbp")
    if ballpark is not None:
        incl = pricing.get("price_incl_vat")
        parts.append(
            f"Approx £{ballpark} including VAT"
            if incl
            else f"Approx £{ballpark}+VAT"
        )
    for item in (pricing.get("line_items") or [])[:6]:
        if isinstance(item, dict) and item.get("item_name"):
            parts.append(str(item["item_name"]))
    return "; ".join(parts)


def _availability_summary_for_llm(avail: dict[str, Any] | None) -> str:
    if not avail:
        return ""
    parts: list[str] = []
    primary = avail.get("primary") or {}
    if primary.get("date"):
        slots = primary.get("available_time_slots") or []
        parts.append(
            f"{primary['date']}: "
            + (", ".join(slots) if slots else "slots unknown")
        )
    for alt in (avail.get("alternatives") or [])[:4]:
        if isinstance(alt, dict) and alt.get("date"):
            slots = alt.get("available_time_slots") or []
            parts.append(
                f"{alt['date']}: " + (", ".join(slots) if slots else "slots unknown")
            )
    time_pref = avail.get("customer_time_preference")
    if time_pref:
        parts.append(f"Customer time preference: {time_pref}")
    return "; ".join(parts)


def _return_quote_replies(
    result: dict[str, Any],
    replies: list[str],
    *,
    customer_name: str,
    subject: str = "",
    thread: list[dict[str, Any]] | None = None,
    content_main: str = "",
    item_description: str = "",
    pricing: dict[str, Any] | None = None,
) -> dict:
    """Return rule-based suggestions plus a complementary companion (rules or LLM)."""
    draft_source = "rules"
    rules_companion = _rules_companion_reply(
        result, customer_name, item_description=item_description
    )
    companion = rules_companion
    llm_err: str | None = None
    if not companion:
        companion, llm_err = llm_suggest_quote_companion(
            customer_name=customer_name,
            phase=str(result.get("phase") or ""),
            missing_slots=list(result.get("missing_slots") or []),
            reason=str(result.get("reason") or ""),
            subject=subject,
            thread_summary=_format_thread_summary_for_llm(thread, content_main),
            standard_replies=replies,
            item_description=item_description,
            pricing_summary=_pricing_summary_for_llm(pricing or result.get("pricing")),
            availability_summary=_availability_summary_for_llm(result.get("availability")),
        )
    out = list(replies)
    if companion:
        out.append(companion)
        draft_source = "rules+supplement" if rules_companion else "rules+llm"
    elif llm_err and is_llm_suggest_enabled():
        result["llm_companion_error"] = llm_err
    return return_replies(result, out, draft_source=draft_source)


def _soften_unpriced_placeholders(replies: list[str]) -> list[str]:
    """Replace £[PRICE]+VAT when catalogue lookup could not produce a ballpark."""
    replacement = (
        "I will confirm the exact price once I have clear photos of the items"
    )
    out: list[str] = []
    for reply in replies:
        text = reply or ""
        if _PRICE_PLACEHOLDER_RE.search(text) or _PRICE_INCL_PLACEHOLDER_RE.search(text):
            text = _PRICE_PLACEHOLDER_RE.sub(replacement, text)
            text = _PRICE_INCL_PLACEHOLDER_RE.sub(replacement, text)
        out.append(text)
    return out


def _attach_pricing(
    result: dict,
    replies: list[str],
    content_main: str,
    *,
    collection_side: str | None = None,
    needs_dismantling: bool = False,
    pricing: dict[str, Any] | None = None,
    skip_price_in_replies: bool = False,
    customer_name: str = "there",
    subject: str = "",
    thread: list[dict[str, Any]] | None = None,
    item_description: str = "",
) -> dict:
    if pricing is None:
        pricing = lookup_prices_for_text(
            content_main,
            collection_side=collection_side,
            needs_dismantling=needs_dismantling,
        )
    if pricing:
        result["pricing"] = {
            "ballpark_gbp": pricing.get("ballpark_gbp"),
            "final_gbp": pricing.get("final_gbp"),
            "price_type": pricing.get("price_type"),
            "line_items": pricing.get("line_items") or [],
            "collection_side": pricing.get("collection_side"),
            "inside_surcharge_applied": pricing.get("inside_surcharge_applied", False),
            "dismantling_surcharge_applied": pricing.get("dismantling_surcharge_applied", False),
            "ambiguous_items": pricing.get("ambiguous_items") or [],
            "unmatched_items": pricing.get("unmatched_items") or [],
            "partial_quote": pricing.get("partial_quote", False),
        }
        if (
            not skip_price_in_replies
            and pricing.get("ballpark_gbp") is not None
            and replies
        ):
            has_placeholder = any(
                _PRICE_PLACEHOLDER_RE.search(r or "")
                or _PRICE_INCL_PLACEHOLDER_RE.search(r or "")
                for r in replies
            )
            if has_placeholder:
                replies = _substitute_reply_prices(replies, pricing)
            else:
                idx = _pick_pricing_reply_index(replies)
                if idx is not None:
                    priced = apply_pricing_to_suggested_replies([replies[idx]], pricing)
                    replies = (
                        replies[:idx]
                        + [_dedupe_inserted_pricing(priced[0])]
                        + replies[idx + 1 :]
                    )
                elif pricing.get("price_incl_vat"):
                    replies = _substitute_reply_prices(replies, pricing)
        elif not skip_price_in_replies and replies and any(
            _PRICE_PLACEHOLDER_RE.search(r or "")
            or _PRICE_INCL_PLACEHOLDER_RE.search(r or "")
            for r in replies
        ):
            replies = _soften_unpriced_placeholders(replies)
    elif replies and any(
        _PRICE_PLACEHOLDER_RE.search(r or "") or _PRICE_INCL_PLACEHOLDER_RE.search(r or "")
        for r in replies
    ):
        replies = _soften_unpriced_placeholders(replies)
    return _return_quote_replies(
        result,
        replies,
        customer_name=customer_name,
        subject=subject,
        thread=thread,
        content_main=content_main,
        item_description=item_description,
        pricing=pricing,
    )


def suggest_quote_reply(
    *,
    content_main: str,
    subject: str = "",
    from_header: str = "",
    parsed_form: dict[str, str] | None = None,
    attachments: list[dict[str, Any]] | None = None,
    thread: list[dict[str, Any]] | None = None,
    greeting_name: str = "there",
) -> dict:
    form = _merge_form(parsed_form, thread, content_main, from_header)
    name = quote_greeting_name(form) or greeting_name

    form_body = _form_body_from_context(thread, content_main)
    intake_main = form_body or content_main
    follow_up = _is_quote_follow_up(thread, content_main, from_header)

    if follow_up:
        collected = _collect_slots(
            thread=thread,
            content_main=content_main,
            from_header=from_header,
            form=form,
            attachments=attachments,
            subject=subject,
        )
        comments = _resolve_item_comments(form, thread, content_main, from_header)
        pipeline_ctx = _build_booking_pipeline_context(
            thread=thread,
            content_main=content_main,
            from_header=from_header,
            comments=comments,
        )
        missing = _follow_up_missing_slots(collected, pipeline_ctx)
        replies, availability_info = _follow_up_variants(
            name,
            form,
            collected,
            missing,
            thread=thread,
            content_main=content_main,
            from_header=from_header,
            pipeline_ctx=pipeline_ctx,
            latest_message=pipeline_ctx.latest_customer_message,
        )
        pricing_text = _pricing_source_text(thread, content_main, form, from_header)
        price_side, needs_dismantling = _pricing_access_context(
            form,
            content_main,
            thread,
            collected=collected,
            from_header=from_header,
        )
        ready_for_booking = _is_ready_for_availability_offer(collected, pipeline_ctx)
        ready_for_confirmation = _is_ready_for_booking_confirmation(
            collected, pipeline_ctx
        )
        booking_requested = (
            "booking_availability" in missing
            or customer_asks_availability(pipeline_ctx.thread_customer_text)
            or _customer_wants_to_proceed(pipeline_ctx.thread_customer_text)
        )
        result_payload: dict[str, Any] = {
                "category": "quote",
                "phase": (
                    "booking_confirmation"
                    if ready_for_confirmation and not missing
                    else "first_contact"
                    if _is_first_customer_inquiry(thread, content_main, from_header)
                    else "follow_up"
                    if missing
                    else "ready_for_staff"
                ),
                "missing_slots": missing,
                "reason": (
                    "All booking details collected — confirm collection and send payment link"
                    if ready_for_confirmation and not missing
                    else "First contact — price, earliest availability if asked, photos and contact details"
                    if _is_first_customer_inquiry(thread, content_main, from_header)
                    and missing
                    else "Follow-up — offer availability and collect remaining contact/location details"
                    if ready_for_booking
                    else "Follow-up — customer asked for collection availability; offer slots and gather booking details"
                    if booking_requested and collected.lwm_already_quoted_price
                    else "Follow-up — only asking for details not yet provided in the thread"
                    if missing
                    else "Follow-up — weight, lift, and location collected from thread"
                ),
                "customer_name": name,
                "is_follow_up": not _is_first_customer_inquiry(
                    thread, content_main, from_header
                ),
                "is_first_contact": _is_first_customer_inquiry(
                    thread, content_main, from_header
                ),
                "has_photos": collected.has_photos,
                "has_item_description": collected.has_item_description,
                "item_description_source": collected.item_description_source,
                "has_collection_location": collected.has_collection_location,
                "alley_inside": collected.alley_inside,
                "access_kind": (collected.access.access_kind if collected.access else None),
                "customer_location_phrase": (
                    collected.access.customer_phrase if collected.access else None
                ),
                "collection_side": (
                    collected.access.collection_side if collected.access else None
                ),
                "is_phone_quote": collected.is_phone_quote,
                "photos_via_phone": collected.photos_via_phone,
                "language": "en-GB",
                "conversation_stage": (
                    "awaiting_intake"
                    if not collected.has_item_description
                    else "ready_to_confirm"
                    if ready_for_confirmation and not missing
                    else "ready_for_booking"
                    if ready_for_booking
                    else "awaiting_booking_details"
                    if booking_requested and collected.lwm_already_quoted_price
                    else "awaiting_location"
                    if "collection_location" in missing
                    else "awaiting_weight_lift"
                    if "item_weight" in missing or "lift_requirement" in missing
                    else "awaiting_booking_details"
                    if missing
                    else "ready_for_booking"
                ),
                "lift_answer": collected.lift_answer,
                "weight_kg": collected.weight_kg,
                "weight_vague": collected.weight_vague,
        }
        if availability_info:
            result_payload["availability"] = availability_info
        pricing = lookup_prices_for_text(
            pricing_text,
            collection_side=price_side,
            needs_dismantling=needs_dismantling,
        )
        thread_quoted = _lwm_quoted_price_gbp(_lwm_reply_bodies(thread))
        price_incl_vat = _lwm_price_is_incl_vat(_lwm_reply_bodies(thread))
        if collected.lwm_already_quoted_price and thread_quoted is not None:
            pricing = {
                **(pricing or {}),
                "ballpark_gbp": thread_quoted,
                "final_gbp": thread_quoted,
                "price_type": "thread_quote",
                "price_incl_vat": price_incl_vat,
                "line_items": (pricing or {}).get("line_items") or [],
            }
        elif not pricing or pricing.get("ballpark_gbp") is None:
            if thread_quoted is not None:
                pricing = {
                    **(pricing or {}),
                    "ballpark_gbp": thread_quoted,
                    "final_gbp": thread_quoted,
                    "price_type": "thread_quote",
                    "line_items": (pricing or {}).get("line_items") or [],
                }
        has_price_placeholder = any(
            _PRICE_PLACEHOLDER_RE.search(r or "")
            or _PRICE_INCL_PLACEHOLDER_RE.search(r or "")
            for r in replies
        )
        return _attach_pricing(
            result_payload,
            replies,
            pricing_text,
            collection_side=price_side,
            needs_dismantling=needs_dismantling,
            pricing=pricing,
            skip_price_in_replies=(
                collected.lwm_already_quoted_price
                and not ready_for_booking
                and not has_price_placeholder
            ),
            customer_name=name,
            subject=subject,
            thread=thread,
            item_description=comments,
        )

    has_photos = _has_customer_photos(intake_main, attachments)
    has_description = _meaningful_comments(intake_main, form)
    missing = _missing_standard_fields(
        form, intake_main, from_header, has_photos=has_photos
    )

    if missing:
        return _return_quote_replies(
            {
                "category": "quote",
                "phase": "missing_fields",
                "missing_slots": missing,
                "reason": "Customer quotation is missing required form fields",
                "customer_name": name,
                "has_photos": has_photos,
                "has_item_description": has_description,
            },
            [_variant_missing_fields(name, missing)],
            customer_name=name,
            subject=subject,
            thread=thread,
            content_main=intake_main,
        )

    price_side, needs_dismantling = _pricing_access_context(
        form, intake_main, thread, from_header=from_header
    )

    if has_photos and not has_description:
        return _attach_pricing(
            {
                "category": "quote",
                "phase": "photos_need_list",
                "missing_slots": ["item_list_in_comments"],
                "reason": "Photos provided without item description or list in comments",
                "customer_name": name,
                "has_photos": True,
                "has_item_description": False,
            },
            [_variant_photos_need_list(name)],
            content_main,
            collection_side=price_side,
            needs_dismantling=needs_dismantling,
            customer_name=name,
            subject=subject,
            thread=thread,
        )

    if has_description and not has_photos:
        pricing = lookup_prices_for_text(
            intake_main,
            collection_side=price_side,
            needs_dismantling=needs_dismantling,
        )
        missing_slots = ["item_photos"]
        replies = _list_need_photos_variants(name, intake_main, form, pricing)
        return _attach_pricing(
            {
                "category": "quote",
                "phase": "list_need_photos",
                "missing_slots": missing_slots,
                "reason": "Item list or description provided without photos",
                "customer_name": name,
                "has_photos": False,
                "has_item_description": True,
                "is_first_contact": True,
                "is_follow_up": False,
            },
            replies,
            intake_main,
            collection_side=price_side,
            needs_dismantling=needs_dismantling,
            pricing=pricing,
            customer_name=name,
            subject=subject,
            thread=thread,
            item_description=(form.get("comments") or _comments_text(intake_main) or ""),
        )

    replies, follow_missing = _intake_complete_variants(
        name, intake_main, form, has_photos=has_photos
    )
    return _attach_pricing(
        {
            "category": "quote",
            "phase": "need_weight_and_access",
            "missing_slots": follow_missing,
            "reason": (
                "Intake complete — approximate price with tailored weight/lift questions"
                + (" and collection location" if "collection_location" in follow_missing else "")
            ),
            "customer_name": name,
            "is_follow_up": False,
            "has_photos": has_photos,
            "has_item_description": has_description,
            "has_collection_location": "collection_location" not in follow_missing,
        },
        replies,
        intake_main,
        collection_side=price_side,
        needs_dismantling=needs_dismantling,
        customer_name=name,
        subject=subject,
        thread=thread,
        item_description=(form.get("comments") or _comments_text(intake_main) or ""),
    )
