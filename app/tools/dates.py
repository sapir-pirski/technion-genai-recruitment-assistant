import re
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS = {
    day: index
    for index, day in enumerate(
        ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    )
}


def parse_date(text: str | None, reference: datetime, timezone: str) -> date | None:
    if not text:
        return None
    current = reference.astimezone(ZoneInfo(timezone)).date()
    value = text.strip().lower()
    if value == "today":
        return current
    if value == "tomorrow":
        return current + timedelta(days=1)
    try:
        return date.fromisoformat(value)
    except ValueError:
        pass
    match = re.fullmatch(r"(?:(next|this)\s+)?(" + "|".join(WEEKDAYS) + r")", value)
    if match:
        prefix, weekday = match.groups()
        delta = (WEEKDAYS[weekday] - current.weekday()) % 7
        # 'next Friday' means Friday of the following ISO calendar week.
        if prefix == "next":
            delta = 7 - current.weekday() + WEEKDAYS[weekday]
        return current + timedelta(days=delta)
    for pattern in ("%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value.replace(",", ""), pattern).date()
        except ValueError:
            continue
    raise ValueError("Please give an exact date (YYYY-MM-DD), today, tomorrow, or a weekday.")


def parse_time(text: str | None) -> time | None:
    if not text:
        return None
    value = re.sub(r"\s+", "", text.lower()).replace(".", "")
    for pattern in ("%I%p", "%I:%M%p", "%H:%M"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            continue
    raise ValueError("Please include AM/PM or use a 24-hour time such as 15:00.")


def local_instant(day: date, at: time, timezone: str) -> datetime:
    zone = ZoneInfo(timezone)
    naive = datetime.combine(day, at)
    first = naive.replace(tzinfo=zone, fold=0)
    second = naive.replace(tzinfo=zone, fold=1)
    if first.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != naive:
        raise ValueError(
            "That local time does not exist because of a clock change. Choose another time."
        )
    if first.utcoffset() != second.utcoffset():
        raise ValueError(
            "That local time occurs twice because of a clock change. Choose another time."
        )
    return first
