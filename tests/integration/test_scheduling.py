import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.config import Settings
from app.graph.builder import build_graph
from app.ingestion.models import DEFAULT_JOB_ID
from app.ingestion.registry import JobRegistry
from app.main import main
from app.storage.models import Booking, Slot, TraceSpan
from app.storage.schedule_repository import BookingConflict
from app.tools.scheduling import scheduling_tool


def test_booking_is_idempotent_and_removes_availability(repository, now):
    slots = repository.available(now, now + timedelta(days=1), "UTC")
    assert [slot["local_time"][11:16] for slot in slots] == ["09:00", "10:00", "11:00"]
    first = repository.book(slots[0]["id"], "alice", "request-1", now, "UTC")
    again = repository.book(slots[0]["id"], "alice", "request-1", now, "UTC")
    assert first == again
    assert len(repository.available(now, now + timedelta(days=1), "UTC")) == 2

    with pytest.raises(BookingConflict):
        repository.book(slots[1]["id"], "alice", "request-2", now, "UTC")


def test_two_candidates_cannot_book_same_slot(repository, now):
    slot = repository.available(now, now + timedelta(days=1), "UTC")[0]

    def reserve(candidate):
        try:
            repository.book(slot["id"], candidate, candidate, now, "UTC")
            return True
        except BookingConflict:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(reserve, ["alice", "bob"])) == [False, True]


def test_past_slot_cannot_be_booked(repository, now):
    slot = repository.available(now, now + timedelta(days=1), "UTC")[0]
    with pytest.raises(BookingConflict):
        repository.book(slot["id"], "alice", "r1", now + timedelta(days=1), "UTC")


def test_partial_options_are_filled_without_relaxing_hard_constraints(
    repository, now, calendar_request
):
    with Session(repository.engine) as session, session.begin():
        session.execute(delete(Slot))
        for day, hour in [(21, 9), (21, 10), (22, 9)]:
            session.add(Slot(starts_at=datetime(2026, 10, day, hour), available=True))
    state = {
        "conversation_id": "partial",
        "request_id": "one",
        "timezone": "UTC",
        "reference_time": now.isoformat(),
        "messages": [{"role": "user", "content": "Any appointments tomorrow?"}],
    }
    request = {**calendar_request, "date_text": "tomorrow"}
    result = scheduling_tool(repository, state).invoke(request)
    assert len(result["slots"]) == 3 and sum(s["is_alternative"] for s in result["slots"]) == 1
    assert len({s["id"] for s in result["slots"]}) == 3
    state.update(
        schedule_preferences=result["preferences"],
        messages=[{"role": "user", "content": "Only tomorrow works."}],
    )
    request["time_text"] = "09:00"  # An offered time is not a newly stated preference.
    request.update(accepts_slot=True, selected_slot_id=result["slots"][0]["id"])
    strict = scheduling_tool(repository, state).invoke(request)
    assert len(strict["slots"]) == 2 and all(
        s["starts_at"].startswith("2026-10-21") for s in strict["slots"]
    )
    state.update(
        schedule_preferences=strict["preferences"],
        messages=[{"role": "user", "content": "Any day works."}],
    )
    cleared = scheduling_tool(repository, state).invoke(calendar_request)
    assert len(cleared["slots"]) == 3 and not cleared["preferences"].get("date")
    assert not cleared["preferences"]["required_fields"]


def test_relative_preference_survives_checkpoint_and_week_boundary(service, calendar_request):
    sunday = datetime(2030, 1, 6, 23, tzinfo=UTC)
    with Session(service.repository.engine) as session, session.begin():
        session.execute(delete(Slot))
        for day in (11, 18):
            for hour in (9, 10, 11):
                session.add(Slot(starts_at=datetime(2030, 1, day, hour), available=True))
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    calendar_request["date_text"] = "next Friday"
    service.turn("week-boundary", "Only next Friday works.", reference_time=sunday)
    prefs = service.state("week-boundary")["schedule_preferences"]
    assert prefs["date"] == "2030-01-11" and prefs["date_reference_time"] == sunday.isoformat()
    service.graph = build_graph(
        service.backend,
        service.repository,
        service.retriever,
        service.graph.checkpointer,
        service.profiles,
    )
    calendar_request["period"] = "morning"
    calendar_request["time_text"] = "09:00"  # Model repeats an offered time; retain all mornings.
    result = service.turn(
        "week-boundary", "Morning would be better.", reference_time=sunday + timedelta(hours=2)
    )
    assert len(result.offered_slots) == 3 and all(
        s["starts_at"].startswith("2030-01-11") for s in result.offered_slots
    )
    assert (
        service.state("week-boundary")["schedule_preferences"]["date_reference_time"]
        == sunday.isoformat()
    )
    calendar_request["date_text"] = "2030-01-11"  # Model repeats an old canonical date.
    changed = service.turn(
        "week-boundary",
        "Actually next Friday instead.",
        reference_time=sunday + timedelta(hours=2, seconds=1),
    )
    assert all(s["starts_at"].startswith("2030-01-18") for s in changed.offered_slots)


@pytest.fixture
def calendar_repository(repository):
    registry = JobRegistry(Settings(_env_file=None))
    with Session(repository.engine) as session, session.begin():
        session.execute(update(Slot).values(duration_minutes=30))
    yield repository
    registry.close()


def test_add_approved_availability_is_atomic_and_preserves_bookings(calendar_repository, now):
    repo = calendar_repository
    original = repo.available(now, None, "UTC")[0]
    booking = repo.book(original["id"], "historical", "historical", now, "UTC")
    starts = [now + timedelta(days=40, hours=1), now + timedelta(days=40, hours=2)]
    preview = repo.add_availability(starts, 30, "UTC", job_id=DEFAULT_JOB_ID, now=now, dry_run=True)
    assert preview["slots_added"] == 0 and preview["slots_planned"] == 2
    result = repo.add_availability(starts, 30, "UTC", job_id=DEFAULT_JOB_ID, now=now)
    assert result["slots_added"] == 2
    for rejected in (
        [starts[0]],
        [starts[0] + timedelta(minutes=15), starts[1] + timedelta(days=1)],
    ):
        with pytest.raises(ValueError, match="duplicates or overlaps"):
            repo.add_availability(rejected, 30, "UTC", job_id=DEFAULT_JOB_ID, now=now)
    assert repo.get_booking("historical", "UTC") == booking
    with Session(repo.engine) as session:
        assert session.scalar(select(func.count()).select_from(Slot)) == 7
        assert session.scalar(select(func.count()).select_from(Booking)) == 1


def test_concurrent_calendar_writers_cannot_insert_overlapping_slots(calendar_repository, now):
    def add(offset):
        try:
            calendar_repository.add_availability(
                [now + timedelta(days=2, minutes=offset)], 30, "UTC", job_id=DEFAULT_JOB_ID, now=now
            )
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(add, (0, 15))) == [False, True]


def test_sql_time_filters_run_before_limit_and_use_candidate_timezone(calendar_repository, now):
    repo = calendar_repository
    starts = [now + timedelta(days=1, hours=i) for i in range(5)]
    repo.add_availability(starts, 30, "UTC", job_id=DEFAULT_JOB_ID, now=now)
    slots = repo.available(now, None, "Asia/Jerusalem", period="afternoon", limit=3)
    assert len(slots) == 3
    assert all(datetime.fromisoformat(slot["local_time"]).hour >= 12 for slot in slots)
    exact = repo.available(
        now, None, "Asia/Jerusalem", exact_time=datetime.strptime("14:00", "%H:%M").time()
    )
    assert len(exact) == 2
    assert all(slot["local_time"][11:16] == "14:00" for slot in exact)


def test_calendar_cli_preview_add_and_status_use_storage_without_models(
    calendar_repository, now, monkeypatch, capsys
):
    monkeypatch.setattr("app.calendar.datetime", Mock(now=Mock(return_value=now)))
    arguments = [
        "recruitment",
        "calendar",
        "add",
        "--job-id",
        DEFAULT_JOB_ID,
        "--from-date",
        "2026-11-18",
        "--to-date",
        "2026-11-18",
        "--weekdays",
        "wed",
        "--start-time",
        "09:00",
        "--end-time",
        "10:00",
        "--timezone",
        "Asia/Jerusalem",
        "--duration-minutes",
        "30",
    ]
    monkeypatch.setattr("sys.argv", arguments + ["--dry-run"])
    main()
    assert json.loads(capsys.readouterr().out)["slots_added"] == 0
    monkeypatch.setattr("sys.argv", arguments)
    main()
    assert json.loads(capsys.readouterr().out)["slots_added"] == 2
    monkeypatch.setattr(
        "sys.argv",
        [
            "recruitment",
            "calendar",
            "status",
            "--job-id",
            DEFAULT_JOB_ID,
            "--timezone",
            "Asia/Jerusalem",
        ],
    )
    main()
    status = json.loads(capsys.readouterr().out)
    assert status["last_configured_date"] == "2026-11-18" and status["future_available"] == 5
    monkeypatch.setattr(
        "app.calendar.datetime", Mock(now=Mock(return_value=now + timedelta(days=31)))
    )
    main()
    expired = json.loads(capsys.readouterr().out)
    assert expired["status"] == "expired" and expired["future_available"] == 0
    assert expired["past_unbooked"] == 5 and expired["next_slots"] == []
    with Session(calendar_repository.engine) as session:
        spans = list(session.scalars(select(TraceSpan).where(TraceSpan.component == "calendar")))
        assert any(
            item.operation == "add_availability" and item.attributes.get("outcome") == "added"
            for item in spans
        )
        assert all(
            item.model is None and item.input_tokens is None and item.conversation_id is None
            for item in spans
        )


def test_calendar_rejects_past_additions_without_mutating_slots(calendar_repository, now):
    repo = calendar_repository
    with pytest.raises(ValueError):
        repo.add_availability(
            [now - timedelta(minutes=1)], 30, "UTC", job_id=DEFAULT_JOB_ID, now=now
        )
    with Session(repo.engine) as session:
        assert session.scalar(select(func.count()).select_from(Slot)) == 5
