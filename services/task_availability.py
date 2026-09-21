"""Fetch collection availability from LWM tasks API for booking replies."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any
import os 
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = (os.getenv("API_BASE_URL") or "").strip().rstrip("/")
AVAILABILITY_PATH = "/api/tasks/availability"
_FETCH_TIMEOUT_SEC = 15

_WEEKDAY_NAMES: dict[str, int] = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "tues": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}

_MONTH_NAMES: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)


@dataclass
class DayAvailability:
    date_iso: str
    date_label: str
    available: bool
    available_time_slots: list[str] = field(default_factory=list)
    blocked_time_slots: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    suggested_days: list[Any] = field(default_factory=list)
    message: str = ""


@dataclass
class BookingAvailabilityPlan:
    """Resolved availability copy for a booking reply."""

    offer_line: str
    checked_dates: list[str] = field(default_factory=list)
    primary: DayAvailability | None = None
    alternatives: list[DayAvailability] = field(default_factory=list)
    customer_preference: str | None = None
    time_preference: str | None = None


def fetch_task_availability(date_iso: str) -> dict[str, Any] | None:
    if not API_BASE_URL:
        return None
    q = urllib.parse.urlencode({"date": date_iso.strip()})
    url = f"{API_BASE_URL}{AVAILABILITY_PATH}?{q}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_SEC) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data if isinstance(data, dict) else None
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError):
        return None


def _format_date_label(d: date, *, today: date) -> str:
    tomorrow = today + timedelta(days=1)
    pretty = f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"
    if d == tomorrow:
        return f"tomorrow ({pretty})"
    return pretty


def _weekday_date(today: date, weekday: int, *, force_next_week: bool = False) -> date:
    """
    Next occurrence of weekday from today.

    If today is that weekday, returns 7 days ahead.
    force_next_week=True skips the upcoming occurrence (use for "Tuesday next week").
    """
    delta = (weekday - today.weekday()) % 7
    if delta == 0:
        delta = 7
    if force_next_week:
        delta += 7
    return today + timedelta(days=delta)


def _resolve_calendar_date(
    day: int,
    month: int,
    *,
    year: int | None = None,
    today: date,
) -> date | None:
    year = year or today.year
    try:
        d = date(year, month, day)
    except ValueError:
        return None
    if year == today.year and d < today:
        try:
            d = date(year + 1, month, day)
        except ValueError:
            return None
    return d


def extract_customer_preferred_dates(
    text: str,
    *,
    today: date | None = None,
) -> list[date]:
    """Dates explicitly mentioned by the customer — no default fallback."""
    today = today or date.today()
    low = (text or "").lower()
    found: list[date] = []
    seen: set[str] = set()

    def _add(d: date) -> None:
        key = d.isoformat()
        if key not in seen:
            seen.add(key)
            found.append(d)

    if re.search(r"\btoday\b", low):
        _add(today)
    if re.search(r"\btomorrow\b", low):
        _add(today + timedelta(days=1))

    for name, wd in _WEEKDAY_NAMES.items():
        # "Tuesday next week" / "next week's Tuesday" → week after the upcoming one
        if re.search(
            rf"\b(?:{re.escape(name)}\s+next\s+week|next\s+week(?:'s)?\s+{re.escape(name)})\b",
            low,
        ):
            _add(_weekday_date(today, wd, force_next_week=True))
        # "next Tuesday" / "Tuesday" → the upcoming Tuesday (or +7 if today is Tuesday)
        elif re.search(rf"\b(?:next\s+)?{re.escape(name)}\b", low):
            _add(_weekday_date(today, wd, force_next_week=False))

    for m in re.finditer(r"\b(20\d{2}-\d{2}-\d{2})\b", text or ""):
        try:
            _add(date.fromisoformat(m.group(1)))
        except ValueError:
            pass

    for m in re.finditer(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](20\d{2})\b", text or ""):
        try:
            _add(date(int(m.group(3)), int(m.group(2)), int(m.group(1))))
        except ValueError:
            pass

    for m in re.finditer(
        rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_PATTERN})(?:\s+(20\d{{2}}))?\b",
        low,
    ):
        month = _MONTH_NAMES.get(m.group(2).replace(".", ""), 0)
        if not month:
            continue
        year = int(m.group(3)) if m.group(3) else None
        resolved = _resolve_calendar_date(
            int(m.group(1)), month, year=year, today=today
        )
        if resolved:
            _add(resolved)

    for m in re.finditer(
        rf"\b({_MONTH_PATTERN})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(20\d{{2}}))?\b",
        low,
    ):
        month = _MONTH_NAMES.get(m.group(1).replace(".", ""), 0)
        if not month:
            continue
        year = int(m.group(3)) if m.group(3) else None
        resolved = _resolve_calendar_date(
            int(m.group(2)), month, year=year, today=today
        )
        if resolved:
            _add(resolved)

    if re.search(r"\bthis week\b", low):
        d = today + timedelta(days=1)
        for _ in range(7):
            _add(d)
            d += timedelta(days=1)

    return found


def customer_states_preferred_date_or_slot(text: str) -> bool:
    """Customer named a calendar date and/or morning/afternoon preference."""
    body = (text or "").strip()
    if not body:
        return False
    if extract_customer_preferred_dates(body):
        return True
    low = body.lower()
    if re.search(r"\b(?:morning|afternoon|am|pm)\b", low) and re.search(
        rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH_PATTERN}|{_MONTH_PATTERN}\s+\d{{1,2}})\b",
        low,
    ):
        return True
    return bool(
        re.search(
            r"\b(?:"
            r"morning slot|afternoon slot|morning collection|afternoon collection|"
            r"available (?:on|for)|collect(?:ion)? on|book(?:ing)? (?:for|on)"
            r")\b",
            low,
        )
        and re.search(rf"\b(?:\d{{1,2}}|{_MONTH_PATTERN})\b", low)
    )


def parse_customer_preferred_dates(text: str, *, today: date | None = None) -> list[date]:
    """Extract dates the customer mentioned; default to tomorrow only when none found."""
    today = today or date.today()
    found = extract_customer_preferred_dates(text, today=today)
    if not found:
        found.append(today + timedelta(days=1))
    return found


def customer_asks_earliest_availability(text: str) -> bool:
    low = (text or "").lower()
    return bool(
        re.search(
            r"\b(?:"
            r"earliest|soonest|as soon as possible|asap|first available|how soon|"
            r"when(?:'s| is) the earliest|what(?:'s| is) the earliest|"
            r"when can you do|when can you collect|when can you come|"
            r"when would you be able|what(?:'s| is) the soonest"
            r")\b",
            low,
        )
    )


def customer_asks_latest_slot(text: str) -> bool:
    """Customer asking for the latest/last available time slot (late in the day)."""
    low = (text or "").lower()
    return bool(
        re.search(r"\b(?:latest|last)\b.{0,40}\b(?:slot|time|collection)s?\b", low)
        or re.search(r"\b(?:slot|time|collection)s?\b.{0,40}\b(?:latest|last)\b", low)
    )


def customer_asks_slot_options(text: str) -> bool:
    """Customer is asking which slots/days are available — not confirming one yet."""
    low = (text or "").strip().lower()
    if not low:
        return False
    if customer_asks_latest_slot(text):
        return True
    if re.search(r"\bother days?(?:\s+of\s+the\s+week)?\b", low):
        return True
    if re.search(r"\b(?:what|which|earliest).{0,40}\b(?:slot|time)s?\b", low):
        return True
    return False


def customer_confirmed_booking_slot(text: str) -> bool:
    """Customer chose or accepted a specific date/slot — not merely asking what's available."""
    body = (text or "").strip()
    if not body:
        return False
    if (
        customer_asks_availability(body)
        or customer_asks_latest_slot(body)
        or customer_asks_slot_options(body)
    ):
        return False
    if parse_customer_api_time_slot(body):
        return True
    if customer_states_preferred_date_or_slot(body) and not customer_asks_availability(body):
        return True
    return bool(
        re.search(
            r"\b(?:yes|ok|okay|works for me|that works|sounds good|please book|go ahead|proceed)\b",
            body.lower(),
        )
        and (
            extract_customer_preferred_dates(body)
            or parse_scheduling_time_preference(body)
        )
    )


def _strip_leading_greeting(low: str) -> str:
    if re.search(r"^good\s+morning\b", low):
        stripped = re.sub(r"^good\s+morning[,!\s]*", "", low, count=1).strip()
        return stripped or low
    return low


def parse_scheduling_time_preference(text: str) -> str | None:
    """
    Time preference from scheduling intent.
    latest slot → afternoon (12pm–5pm).
    earliest slot (explicit) or earliest availability → morning (7am–12pm).
    """
    body = (text or "").strip()
    if not body:
        return None
    low = _strip_leading_greeting(body.lower())

    if customer_asks_latest_slot(body):
        return "afternoon"

    if re.search(
        r"\b(?:earliest|soonest|first)\b.{0,40}\b(?:slot|time)s?\b",
        low,
    ) or re.search(
        r"\b(?:slot|time)s?\b.{0,40}\b(?:earliest|soonest)\b",
        low,
    ):
        return "morning"

    if customer_asks_earliest_availability(body):
        return "morning"

    return parse_customer_time_preference(body)


def customer_specified_time_slot(text: str) -> bool:
    """True when the customer explicitly named a time slot preference."""
    body = (text or "").strip()
    if not body:
        return False
    if parse_customer_api_time_slot(body):
        return True
    return parse_scheduling_time_preference(body) is not None


def customer_asks_availability(text: str) -> bool:
    low = (text or "").lower()
    if customer_asks_earliest_availability(text):
        return True
    if customer_asks_latest_slot(text):
        return True
    if customer_asks_slot_options(text):
        return True
    if customer_states_preferred_date_or_slot(text):
        return True
    return bool(
        re.search(
            r"\b(?:"
            r"what days?|which days?|when can you|what date|availability|"
            r"this week|next week|could work|available dates?|"
            r"what times?|date in mind|cleared this week|collect(?:ion)? this week|"
            r"when works for you|works for you|what works for you|"
            r"available this afternoon|this afternoon or|when (?:would|can) (?:you|we)|"
            r"like to proceed|would like to proceed|please book|book(?:ing)? in|"
            r"collection slots?|time slots?"
            r")\b",
            low,
        )
        or re.search(
            r"\b(?:what|which)\s+days?\b.{0,40}\b(?:work|suit|available)\b",
            low,
        )
        or re.search(r"\bwhen\s+works\b", low)
    )


def parse_customer_time_preference(text: str) -> str | None:
    """Return morning | afternoon | evening | anytime, or None if unspecified."""
    body = (text or "").strip()
    if not body:
        return None
    low = body.lower()
    low = _strip_leading_greeting(low)
    if not low.strip():
        return None

    if re.search(
        r"\b(?:any\s*time|anytime|flexible time|whenever|any slot|whole day)\b",
        low,
    ):
        return "anytime"

    if re.search(
        r"\b(?:"
        r"evening|eve\b|tonight|after work|after\s+5|after\s+five|"
        r"late afternoon|end of (?:the )?day|later in the day"
        r")\b",
        low,
    ):
        return "evening"

    if re.search(
        r"\b(?:"
        r"afternoon|after lunch|post lunch|after noon|p\.m\.|"
        r"afternoon slot|afternoon collection"
        r")\b",
        low,
    ):
        return "afternoon"

    if re.search(
        r"\b(?:"
        r"morning|before lunch|early(?:\s+morning)?|first thing|"
        r"a\.m\.|am slot|morning slot|morning collection"
        r")\b",
        low,
    ):
        return "morning"

    if re.search(r"\bam\b", low) and not re.search(r"\bpm\b", low):
        return "morning"
    if re.search(r"\bpm\b", low) and not re.search(r"\bam\b", low):
        return "afternoon"

    return None


def resolve_customer_time_preference(
    customer_text: str,
    *,
    scheduling_text: str | None = None,
) -> str | None:
    """Prefer the latest scheduling message, then scan customer lines newest-first."""
    if scheduling_text and customer_asks_slot_options(scheduling_text):
        pref = parse_scheduling_time_preference(scheduling_text)
        if pref:
            return pref
        return None
    if scheduling_text:
        pref = parse_scheduling_time_preference(scheduling_text)
        if pref:
            return pref
        pref = parse_customer_time_preference(scheduling_text)
        if pref:
            return pref
    for line in reversed((customer_text or "").splitlines()):
        if customer_asks_slot_options(line):
            pref = parse_scheduling_time_preference(line)
            if pref:
                return pref
            continue
        pref = parse_scheduling_time_preference(line) or parse_customer_time_preference(line)
        if pref:
            return pref
    return parse_scheduling_time_preference(customer_text) or parse_customer_time_preference(
        customer_text
    )


def customer_asks_general_availability(text: str) -> bool:
    low = (text or "").lower()
    if customer_asks_earliest_availability(text):
        return True
    if customer_asks_slot_options(text):
        return True
    if re.search(r"\bthis week\b", low):
        return True
    if re.search(r"\bother days?(?:\s+of\s+the\s+week)?\b", low):
        return True
    if re.search(
        r"\b(?:what|which)\s+days?\b.{0,40}\b(?:work|suit|available|could)\b",
        low,
    ):
        return True
    if re.search(r"\bdate in mind\b", low):
        return True
    return False


def parse_customer_api_time_slot(text: str) -> str | None:
    """Map customer wording to an API time slot: AnyTime | 7am-12pm | 12pm-5pm."""
    body = (text or "").strip()
    if not body:
        return None
    compact = body.lower().replace("–", "-")
    norm = re.sub(r"\s+", "", compact)

    if re.search(r"\bany\s*time\b|\banytime\b", compact):
        return "AnyTime"
    if re.search(r"7am-12pm", norm) or re.search(
        r"7\s*am\s*-\s*12\s*pm", compact
    ):
        return "7am-12pm"
    if re.search(r"12pm-5pm", norm) or re.search(
        r"12\s*pm\s*-\s*5\s*pm", compact
    ):
        return "12pm-5pm"

    pref = parse_customer_time_preference(body)
    if pref == "morning":
        return "7am-12pm"
    if pref in ("afternoon", "evening"):
        return "12pm-5pm"
    if pref == "anytime":
        return "AnyTime"
    return None


def resolve_customer_booking_schedule(
    customer_bodies: list[str],
    *,
    today: date | None = None,
) -> dict[str, str | None]:
    """Merge date and time-slot choices across customer messages (newest wins per field)."""
    today = today or date.today()
    booking_date: date | None = None
    time_slot: str | None = None

    for body in reversed(customer_bodies or []):
        if not time_slot:
            time_slot = parse_customer_api_time_slot(body)
        if not booking_date:
            dates = extract_customer_preferred_dates(body, today=today)
            if dates:
                booking_date = dates[0]

    return {
        "bookingDate": booking_date.isoformat() if booking_date else None,
        "bookingDateLabel": (
            _format_date_label(booking_date, today=today) if booking_date else None
        ),
        "bookingTimeSlot": time_slot,
    }


def _normalize_slot_key(raw: str) -> str:
    return (raw or "").strip().lower().replace("–", "-").replace(" ", "")


def _slot_kind(api_slot: str) -> str | None:
    s = _normalize_slot_key(api_slot)
    if s == "anytime":
        return "anytime"
    if "7am" in s and "12pm" in s:
        return "morning"
    if "12pm" in s and "5pm" in s:
        return "afternoon"
    return None


def _preference_matches_slot(preference: str, api_slot: str) -> bool:
    kind = _slot_kind(api_slot)
    if not kind:
        return False
    if preference == "anytime":
        return kind == "anytime"
    if preference == "morning":
        return kind == "morning"
    if preference in ("afternoon", "evening"):
        return kind == "afternoon"
    return False


def _slots_matching_preference(slots: list[str], preference: str) -> list[str]:
    return [s for s in (slots or []) if _preference_matches_slot(preference, s)]


def _preference_phrase(preference: str) -> str:
    return {
        "morning": "in the morning",
        "afternoon": "in the afternoon",
        "evening": "in the afternoon",
        "anytime": "at any time",
    }.get(preference, "")


def _preference_slot_display(preference: str) -> str:
    return {
        "morning": "7am–12pm",
        "afternoon": "12pm–5pm",
        "evening": "12pm–5pm",
        "anytime": "any time",
    }.get(preference, "")


def _format_all_time_slots(slots: list[str]) -> str:
    """List every available API slot explicitly (any time, 7am–12pm, 12pm–5pm, …)."""
    labels: list[str] = []
    for raw in slots or []:
        s = (raw or "").strip()
        if not s:
            continue
        if s.lower().replace(" ", "") == "anytime":
            labels.append("any time")
        else:
            labels.append(s.replace("-", "–"))
    if not labels:
        return "any time, 7am–12pm or 12pm–5pm"
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f" or {labels[-1]}"


def _format_time_slots(slots: list[str]) -> str:
    return _format_all_time_slots(slots)


def _day_from_api_payload(payload: dict[str, Any], *, today: date) -> DayAvailability:
    iso = str(payload.get("date") or "").strip()
    try:
        d = date.fromisoformat(iso)
        label = _format_date_label(d, today=today)
    except ValueError:
        label = iso or "that day"
    return DayAvailability(
        date_iso=iso,
        date_label=label,
        available=bool(payload.get("available")),
        available_time_slots=list(payload.get("availableTimeSlots") or []),
        blocked_time_slots=list(payload.get("blockedTimeSlots") or []),
        reasons=[str(r) for r in (payload.get("reasons") or []) if r],
        suggested_days=list(payload.get("suggestedDays") or []),
        message=str(payload.get("message") or "").strip(),
    )


def _suggested_dates_from_payload(day: DayAvailability) -> list[str]:
    out: list[str] = []
    for item in day.suggested_days:
        if isinstance(item, str) and item.strip():
            out.append(item.strip()[:10])
        elif isinstance(item, dict):
            raw = item.get("date") or item.get("day")
            if raw:
                out.append(str(raw).strip()[:10])
    return out


def _offer_line_for_available(
    day: DayAvailability,
    *,
    earliest: bool = False,
    time_preference: str | None = None,
) -> str:
    if time_preference:
        matched = _slots_matching_preference(day.available_time_slots, time_preference)
        phrase = _preference_phrase(time_preference)
        slot_label = _preference_slot_display(time_preference)
        if matched:
            if time_preference == "anytime":
                detail = "at any time"
            else:
                detail = f"{phrase} ({slot_label})"
            prefix = (
                f"The earliest we can offer is {day.date_label}"
                if earliest
                else f"We can send a team on {day.date_label}"
            )
            return (
                f"{prefix} — we have availability {detail}. "
                "Please let me know if that works for you."
            )
        alt_slots = _format_all_time_slots(day.available_time_slots)
        unavailable = (
            "a morning slot (7am–12pm)"
            if time_preference == "morning"
            else (
                "an afternoon slot (12pm–5pm)"
                if time_preference in ("afternoon", "evening")
                else "an any-time slot"
            )
        )
        prefix = (
            f"The earliest we can offer is {day.date_label}"
            if earliest
            else f"Unfortunately we do not have {unavailable} available on {day.date_label}"
        )
        if earliest:
            return (
                f"{prefix} — available time slots: {alt_slots}. "
                "Please let me know which time slot works best for you."
            )
        return (
            f"{prefix}. "
            f"We do have availability: {alt_slots}. "
            "Please let me know which time slot works best for you."
        )

    slots = _format_all_time_slots(day.available_time_slots)
    if earliest:
        return (
            f"The earliest we can offer is {day.date_label} — "
            f"available time slots: {slots}. "
            "Please let me know which time slot works best for you."
        )
    return (
        f"We can send a team on {day.date_label} — "
        f"available time slots: {slots}. "
        "Please let me know which time slot works best for you."
    )


def confirmation_offer_line(plan: BookingAvailabilityPlan) -> str:
    """Slot wording for booking confirmation — respects customer time preference when set."""
    if plan.primary and plan.primary.available:
        if plan.time_preference:
            matched = _slots_matching_preference(
                plan.primary.available_time_slots, plan.time_preference
            )
            if matched:
                if plan.time_preference == "anytime":
                    detail = "at any time"
                else:
                    detail = (
                        f"{_preference_phrase(plan.time_preference)} "
                        f"({_preference_slot_display(plan.time_preference)})"
                    )
                return (
                    f"We can send a team on {plan.primary.date_label} — "
                    f"we have availability {detail}."
                )
        slots = _format_all_time_slots(plan.primary.available_time_slots)
        return (
            f"We can send a team on {plan.primary.date_label}. "
            f"Available time slots: {slots}."
        )
    return plan.offer_line


def _offer_line_for_unavailable(
    day: DayAvailability,
    alternatives: list[DayAvailability],
) -> str:
    if alternatives:
        parts: list[str] = []
        for alt in alternatives[:3]:
            slots = _format_time_slots(alt.available_time_slots)
            parts.append(f"{alt.date_label} ({slots})")
        alt_text = "; ".join(parts)
        return (
            f"Unfortunately we are fully booked on {day.date_label}. "
            f"We do have availability on {alt_text}. "
            "Please let me know which day and time slot suit you."
        )
    reason = day.reasons[0] if day.reasons else "that date is fully booked"
    return (
        f"Unfortunately {day.date_label} is not available ({reason}). "
        "Please let me know your preferred date and whether you would prefer "
        "a morning (7am–12pm) or afternoon (12pm–5pm) slot."
    )


COLLECTION_WORKING_DAYS_LINE = "We work Monday to Sunday."


def build_booking_availability_plan(
    customer_text: str,
    *,
    today: date | None = None,
    scheduling_text: str | None = None,
) -> BookingAvailabilityPlan:
    today = today or date.today()
    intent_text = (scheduling_text or "").strip() or customer_text
    time_preference = resolve_customer_time_preference(
        customer_text, scheduling_text=scheduling_text
    )
    if customer_asks_earliest_availability(intent_text):
        time_preference = "morning"
    elif not customer_specified_time_slot(intent_text):
        time_preference = None
    earliest = customer_asks_earliest_availability(intent_text)
    slot_options = customer_asks_slot_options(intent_text)
    general = customer_asks_general_availability(intent_text)
    preferred = parse_customer_preferred_dates(intent_text, today=today)
    if earliest or (general and len(preferred) <= 1):
        preferred = []
        d = today + timedelta(days=1)
        for _ in range(6):
            preferred.append(d)
            d += timedelta(days=1)
    checked: list[str] = []
    alternatives: list[DayAvailability] = []
    primary: DayAvailability | None = None

    for d in preferred[:3]:
        iso = d.isoformat()
        payload = fetch_task_availability(iso)
        checked.append(iso)
        if not payload:
            continue
        day = _day_from_api_payload(payload, today=today)
        if day.available:
            primary = day
            break
        if primary is None:
            primary = day
        for sug_iso in _suggested_dates_from_payload(day)[:4]:
            if sug_iso in checked:
                continue
            sug_payload = fetch_task_availability(sug_iso)
            checked.append(sug_iso)
            if not sug_payload:
                continue
            sug_day = _day_from_api_payload(sug_payload, today=today)
            if sug_day.available:
                alternatives.append(sug_day)

    if general and primary and primary.available:
        week_line = COLLECTION_WORKING_DAYS_LINE
        if slot_options and not customer_specified_time_slot(intent_text):
            primary_line = (
                f"We can send a team on {primary.date_label} — "
                f"available time slots: {_format_all_time_slots(primary.available_time_slots)}. "
                "Please let me know which day and time slot work best for you."
            )
        else:
            primary_line = _offer_line_for_available(
                primary, earliest=earliest, time_preference=time_preference
            )
        offer = f"{week_line}\n\n{primary_line}"
        if alternatives and not earliest:
            alt_bits = [
                f"{a.date_label} ({_format_all_time_slots(a.available_time_slots)})"
                for a in alternatives[:3]
            ]
            if alt_bits:
                offer += f"\n\nWe also have availability on {', '.join(alt_bits)}."
    elif primary and primary.available:
        offer = _offer_line_for_available(
            primary, earliest=earliest, time_preference=time_preference
        )
    elif primary:
        offer = _offer_line_for_unavailable(primary, alternatives)
    else:
        fallback_iso = (today + timedelta(days=1)).isoformat()
        offer = (
            "We can send a team tomorrow — please let me know whether you would prefer "
            "a morning (7am–12pm) or afternoon (12pm–5pm) slot."
        )
        checked.append(fallback_iso)

    pref = None
    low = intent_text.lower()
    if re.search(r"\btomorrow\b", low):
        pref = "tomorrow"
    else:
        for name in _WEEKDAY_NAMES:
            if re.search(rf"\b(?:next\s+)?{re.escape(name)}\b", low):
                pref = name
                break

    return BookingAvailabilityPlan(
        offer_line=offer,
        checked_dates=checked,
        primary=primary,
        alternatives=alternatives,
        customer_preference=pref,
        time_preference=time_preference,
    )


def availability_metadata(plan: BookingAvailabilityPlan) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "checked_dates": plan.checked_dates,
        "customer_preference": plan.customer_preference,
        "customer_time_preference": plan.time_preference,
    }
    if plan.primary:
        meta["primary"] = {
            "date": plan.primary.date_iso,
            "available": plan.primary.available,
            "available_time_slots": plan.primary.available_time_slots,
            "blocked_time_slots": plan.primary.blocked_time_slots,
            "reasons": plan.primary.reasons,
            "suggested_days": plan.primary.suggested_days,
        }
    if plan.alternatives:
        meta["alternatives"] = [
            {
                "date": a.date_iso,
                "available_time_slots": a.available_time_slots,
            }
            for a in plan.alternatives
        ]
    return meta
