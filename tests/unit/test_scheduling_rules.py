from datetime import UTC, date, datetime, time

import pytest

from app.agents.backends.common import explicit_stop
from app.calendar import availability_times
from app.tools.dates import parse_date, parse_time


def window(**changes):
    values = dict(
        start_date=date(2026, 11, 4),
        end_date=date(2026, 11, 17),
        weekdays=["sun", "mon", "tue", "wed", "thu"],
        start_time=time(9),
        end_time=time(17),
        timezone="Asia/Jerusalem",
        duration_minutes=30,
        now=datetime(2026, 11, 1, tzinfo=UTC),
    )
    return availability_times(**{**values, **changes})


def test_approved_calendar_has_160_timezone_aware_appointments():
    starts = window()
    assert len(starts) == 160
    assert starts[0].isoformat() == "2026-11-04T09:00:00+02:00"
    assert starts[-1].isoformat() == "2026-11-17T16:30:00+02:00"
    assert all(start.weekday() in (6, 0, 1, 2, 3) for start in starts)


def test_calendar_rejects_nonpositive_duration():
    with pytest.raises(ValueError):
        window(duration_minutes=0)


@pytest.mark.parametrize(
    "day,start,end,duration",
    [(date(2026, 3, 8), time(2), time(3), 30), (date(2026, 11, 1), time(1), time(2), 30)],
)
def test_calendar_rejects_nonexistent_or_ambiguous_local_times(day, start, end, duration):
    with pytest.raises(ValueError):
        window(
            start_date=day,
            end_date=day,
            weekdays=["sun"],
            start_time=start,
            end_time=end,
            duration_minutes=duration,
            timezone="America/New_York",
            now=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_weekday_policy_uses_message_time(now):
    assert parse_date("this Friday", now, "UTC") == date(2026, 10, 23)
    assert parse_date("next Friday", now, "UTC") == date(2026, 10, 30)
    historical = datetime.fromisoformat("2024-04-03T15:12:00+00:00")
    assert parse_date("next Friday", historical, "UTC") == date(2024, 4, 12)


def test_timezone_changes_reference_day():
    reference = datetime.fromisoformat("2026-10-20T01:00:00+00:00")
    assert parse_date("tomorrow", reference, "America/New_York") == date(2026, 10, 20)


def test_ambiguous_time_requires_clarification():
    with pytest.raises(ValueError):
        parse_time("3")


def test_negated_opt_out():
    for text in (
        "Do not stop texting me",
        "What happens if I accepted another job?",
        "I am not interested in frontend work; I am interested in this Python role.",
        'My old application says "not interested", but I want to continue here.',
    ):
        assert not explicit_stop(text)
