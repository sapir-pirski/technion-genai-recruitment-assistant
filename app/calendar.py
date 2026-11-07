"""Explicit recruiter-approved calendar management; no automatic calendar extension."""

from datetime import UTC, datetime, time, timedelta
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.observability.tracing import TraceRecorder
from app.storage.database import make_engine
from app.storage.schedule_repository import ScheduleRepository
from app.tools.dates import local_instant

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def availability_times(
    *, start_date, end_date, weekdays, start_time, end_time, timezone, duration_minutes, now
):
    ZoneInfo(timezone)
    if not 0 <= (end_date - start_date).days <= 365:
        raise ValueError("Choose an inclusive date range of at most 366 days")
    if not weekdays or not set(weekdays) <= set(WEEKDAYS):
        raise ValueError("Provide working weekdays using mon through sun")
    if (
        any(t.second or t.microsecond or t.tzinfo for t in (start_time, end_time))
        or start_time >= end_time
    ):
        raise ValueError("Working hours must use local HH:MM times within one day")
    if not 1 <= duration_minutes <= 480:
        raise ValueError("Appointment duration must be between 1 and 480 minutes")
    begin = start_time.hour * 60 + start_time.minute
    finish = end_time.hour * 60 + end_time.minute
    if (finish - begin) % duration_minutes:
        raise ValueError("The working window must divide evenly into complete appointments")
    result = []
    day = start_date
    while day <= end_date:
        if WEEKDAYS[day.weekday()] in weekdays:
            for minute in range(begin, finish, duration_minutes):
                start = local_instant(day, time(minute // 60, minute % 60), timezone)
                ending = minute + duration_minutes
                end = local_instant(day, time(ending // 60, ending % 60), timezone)
                if end.astimezone(UTC) - start.astimezone(UTC) != timedelta(
                    minutes=duration_minutes
                ):
                    raise ValueError(
                        "The working window crosses a clock change; choose hours after the transition"
                    )
                if start <= now:
                    raise ValueError("Every appointment must start in the future")
                result.append(start)
        day += timedelta(days=1)
    if not result:
        raise ValueError("No appointments match the selected weekdays and date range")
    if len(result) > 10000:
        raise ValueError("Add at most 10000 appointments per command")
    return result


def manage_calendar(settings, args):
    engine = make_engine(settings.database_url)
    try:
        repository = ScheduleRepository(engine)
        now = datetime.now(UTC)
        timezone = args.timezone or settings.business_timezone
        # Calendar operations need storage only and never initialize a model client.
        with TraceRecorder(engine).trace(
            None,
            str(uuid4()),
            component="calendar",
            operation=args.calendar_action,
            attributes={"job_id": args.job_id, "timezone": timezone},
        ) as item:
            if args.calendar_action == "status":
                result = repository.calendar_status(args.job_id, timezone, now)
                item.attributes.update(
                    last_configured_date=result["last_configured_date"],
                    future_available=result["future_available"],
                )
            else:
                item.attributes.update(
                    from_date=args.from_date.isoformat(),
                    to_date=args.to_date.isoformat(),
                    weekdays=args.weekdays,
                    start_time=args.start_time.isoformat(),
                    end_time=args.end_time.isoformat(),
                    duration_minutes=args.duration_minutes,
                    dry_run=args.dry_run,
                    initiated_by="operator_cli",
                )
                starts = availability_times(
                    start_date=args.from_date,
                    end_date=args.to_date,
                    weekdays=args.weekdays,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    timezone=timezone,
                    duration_minutes=args.duration_minutes,
                    now=now,
                )
                result = repository.add_availability(
                    starts,
                    args.duration_minutes,
                    timezone,
                    job_id=args.job_id,
                    now=now,
                    dry_run=args.dry_run,
                )
            item.attributes["outcome"] = result["status"]
            return result
    except ZoneInfoNotFoundError as exc:
        raise ValueError("Use a valid IANA timezone such as Asia/Jerusalem") from exc
    finally:
        engine.dispose()
