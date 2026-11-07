"""Resolve calendar preferences once and preserve their message-time provenance."""

import re
from datetime import date, time

from app.agents.backends.common import latest_text
from app.observability.logging import log_event
from app.tools.dates import parse_date, parse_time

_DAYS = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTHS = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_DATES = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:next|this)\s+"
    + _DAYS
    + "|"
    + _DAYS
    + r"|today|tomorrow|"
    + _MONTHS
    + r"\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+"
    + _MONTHS
    + r"\s+\d{4})\b",
    re.I,
)
_TIMES = re.compile(r"\b\d{1,2}(?::\d{2})?\s*[ap]\.?m\.?\b|\b\d{1,2}:\d{2}\b", re.I)
_STRICT = re.compile(r"\b(?:only|must|cannot|can't|no other)\b", re.I)


def _key(value):
    return " ".join(value.casefold().split())


def _stated(phrase, text):
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(_key(phrase)) + r"(?!\w)", _key(text)))


def _equivalent(pattern, parser, expected, text):
    for match in pattern.finditer(text):
        try:
            if parser(match[0]) == expected:
                return True
        except ValueError:
            continue
    return False


def resolve_preferences(request, state, now, zone):
    previous = state.get("schedule_preferences", {})
    prefs = dict(previous) if previous.get("timezone", zone) == zone else {}
    required = set(prefs.get("required_fields", []))
    latest = latest_text(state)
    cleared = set()

    def clear(field):
        prefs.pop(field, None)
        required.discard(field)
        for suffix in ("phrase", "reference_time", "request_id"):
            prefs.pop(f"{field}_{suffix}", None)

    if re.search(r"\b(?:no preferences?|reset (?:my )?preferences|start over)\b", latest, re.I):
        cleared.update(("date", "time", "period"))
    if re.search(r"\b(?:any (?:day|date)|no date preference)\b", latest, re.I):
        cleared.add("date")
    if re.search(r"\b(?:any\s?time|no time preference)\b", latest, re.I):
        cleared.update(("time", "period"))
    for field in cleared:
        clear(field)
    changed = set()

    def update(field, value, phrase):
        prefs[field] = value
        prefs[f"{field}_phrase"] = phrase
        prefs[f"{field}_reference_time"] = now.isoformat()
        prefs[f"{field}_request_id"] = state["request_id"]
        required.discard(field)
        changed.add(field)

    if request.date_text:
        # Repeated relative text from older history must never be anchored to now.
        stated = _stated(request.date_text, latest)
        phrase = request.date_text
        if stated:
            value = parse_date(request.date_text, now, zone)
        else:
            try:
                value = parse_date(request.date_text, now, zone)
            except ValueError:
                value = None
            stated = value is not None and _equivalent(
                _DATES, lambda phrase: parse_date(phrase, now, zone), value, latest
            )
        if not stated:
            # A model may repeat an old ISO date even when the candidate explicitly
            # changes a relative date. One unambiguous new source date takes priority.
            dates = {}
            for match in _DATES.finditer(latest):
                try:
                    dates[parse_date(match[0], now, zone)] = match[0]
                except ValueError:
                    continue
            if len(dates) == 1:
                value, phrase = next(iter(dates.items()))
                stated = True
        if stated:
            update("date", value.isoformat(), phrase)
        elif "date" not in prefs and "date" not in cleared:
            log_event(
                "calendar_preference_ignored",
                component="app.calendar",
                stage="date",
                reason_code="not_stated_in_latest_message",
            )
    if request.time_text:
        literal = _stated(request.time_text, latest)
        try:
            value = parse_time(request.time_text)
        except ValueError:
            if literal:
                raise
            value = None
        stated = value is not None and (literal or _equivalent(_TIMES, parse_time, value, latest))
        if stated:
            update("time", value.isoformat(), request.time_text)
            # A new exact time replaces an older period restriction when incompatible.
            if prefs.get("period") and ((value.hour < 12) != (prefs["period"] == "morning")):
                clear("period")
        elif "time" not in prefs and "time" not in cleared:
            log_event(
                "calendar_preference_ignored",
                component="app.calendar",
                stage="time",
                reason_code="not_stated_in_latest_message",
            )
    if request.period and _stated(request.period, latest):
        update("period", request.period, request.period)
        if "time" not in changed:
            clear("time")
    if _STRICT.search(latest):
        required.update(
            changed or (field for field in ("date", "time", "period") if prefs.get(field))
        )
    prefs["required_fields"] = sorted(required)
    prefs["timezone"] = zone
    return (
        date.fromisoformat(prefs["date"]) if prefs.get("date") else None,
        time.fromisoformat(prefs["time"]) if prefs.get("time") else None,
        prefs.get("period"),
        prefs,
    )
