from datetime import timedelta
from unittest.mock import Mock

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.scheduling.models import Booking, Slot
from app.scheduling.repository import ScheduleRepository
from app.storage.database import make_engine
from app.storage.legacy import copy_legacy
from app.storage.models import Conversation, ConversationMessage
from app.workflow.service import ChatService


def snapshot(engine, schema):
    quote = engine.dialect.identifier_preparer.quote
    with engine.connect() as connection:
        return {
            name: list(
                connection.scalars(
                    text(
                        f"SELECT row_to_json(t)::text FROM {quote(schema)}.{quote(name)} t ORDER BY 1"
                    )
                )
            )
            for name in inspect(engine).get_table_names(schema=schema)
        }


@pytest.fixture
def legacy_runtime(backend, now):
    # Simulate a deployed public schema, with real checkpoints and retired data.
    settings = Settings(_env_file=None).model_copy(update={"database_schema": "public"})
    repository = ScheduleRepository(make_engine(settings.database_url, schema="public"))
    repository.initialize()
    with Session(repository.engine) as session, session.begin():
        session.add_all(
            [
                Slot(starts_at=(now + timedelta(hours=i)).replace(tzinfo=None), duration_minutes=30)
                for i in (1, 2, 3)
            ]
        )
        session.add(
            Slot(starts_at=(now + timedelta(hours=8)).replace(tzinfo=None), job_id="legacy-role")
        )
    worker = ChatService(
        settings,
        backend=backend,
        repository=repository,
        retriever=Mock(search=Mock(return_value=[])),
    )
    backend.plan.return_value = backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    offered = worker.turn("booked", "Show interviews", request_id="offer", reference_time=now)
    booked = worker.turn(
        "booked",
        "Confirm",
        request_id="confirm",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    worker.turn("active", "I build Python APIs", request_id="background", reference_time=now)
    with worker.registry.engine.begin() as connection:
        connection.execute(
            text("CREATE TABLE public.candidate_profiles (id text PRIMARY KEY, data jsonb)")
        )
        connection.execute(
            text(
                "INSERT INTO public.candidate_profiles VALUES ('preserve-me', '{\"archived\": true}')"
            )
        )
    try:
        yield worker, booked
    finally:
        worker.close()


def test_legacy_copy_is_atomic_preserves_source_and_resumes_chats(legacy_runtime, backend, now):
    source, booked = legacy_runtime
    settings = Settings(_env_file=None)
    before = snapshot(source.registry.engine, "public")
    dry_run = copy_legacy(settings, dry_run=True)
    assert dry_run["copied_rows"]["interview_bookings"] == 1
    target = make_engine(settings.database_url)
    try:
        with Session(target) as session:
            assert session.scalar(select(func.count()).select_from(Slot)) == 0
            assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 0
        copied = copy_legacy(settings)
        assert copied["copied_rows"]["interview_slots"] == 3
        assert copied["copied_rows"]["conversations"] == 2
        assert copied["copied_rows"]["checkpoints"] > 0
        assert snapshot(source.registry.engine, "public") == before
        calls = list(backend.method_calls)
        worker = ChatService(
            settings, backend=backend, retriever=Mock(search=Mock(return_value=[]))
        )
        try:
            assert (
                worker.turn("booked", "Confirm", request_id="confirm", reference_time=now) == booked
            )
            assert backend.method_calls == calls
            assert worker.state("active")["messages"][0]["content"] == "I build Python APIs"
            reply = worker.turn("active", "I also use SQL", request_id="next", reference_time=now)
            assert not reply.error
            assert any(
                message["content"] == "I build Python APIs"
                for message in worker.state("active")["messages"]
            )
            with Session(target) as session, session.begin():
                assert session.scalar(select(func.count()).select_from(Booking)) == 1
                assert session.scalar(select(func.count()).select_from(Conversation)) == 2
                slot = Slot(starts_at=(now + timedelta(days=1)).replace(tzinfo=None))
                session.add(slot)
                session.flush()
                assert slot.id > 3
        finally:
            worker.close()
        assert snapshot(source.registry.engine, "public") == before
        with pytest.raises(ValueError, match="must be empty"):
            copy_legacy(settings)
        assert snapshot(source.registry.engine, "public") == before
    finally:
        target.dispose()


def test_copy_rejects_identical_schema_and_missing_source_tables():
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="must differ"):
        copy_legacy(settings, source_schema=settings.database_schema)
    with pytest.raises(ValueError, match="legacy conversation/calendar"):
        copy_legacy(settings, source_schema="public")


def test_failed_copy_rolls_back_all_rows_and_preserves_the_source(legacy_runtime, now):
    source, _ = legacy_runtime
    with Session(source.registry.engine) as session, session.begin():
        # The legacy schema permits this inconsistent cross-role booking.
        # Its slot is excluded from the assignment-only copy, so the target FK must reject it.
        session.add(
            Booking(
                id="inconsistent-booking",
                slot_id=4,
                conversation_id="inconsistent",
                request_id="inconsistent",
                created_at=now.replace(tzinfo=None),
            )
        )
    before = snapshot(source.registry.engine, "public")
    with pytest.raises(ValueError, match="target constraints"):
        copy_legacy(Settings(_env_file=None))
    assert snapshot(source.registry.engine, "public") == before
    target = make_engine(source.settings.database_url)
    try:
        with Session(target) as session:
            for model in (Slot, Booking, Conversation, ConversationMessage):
                assert session.scalar(select(func.count()).select_from(model)) == 0
    finally:
        target.dispose()


@pytest.mark.parametrize(
    "schema", ["public", "pg_catalog", "information_schema", "unsafe; DROP SCHEMA public"]
)
def test_runtime_requires_a_dedicated_schema(schema):
    with pytest.raises(ValueError):
        Settings(_env_file=None, database_schema=schema)
