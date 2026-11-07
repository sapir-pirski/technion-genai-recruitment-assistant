import hashlib
import json
import os
import shutil
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock
from uuid import uuid4

import httpx
import psycopg
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from psycopg import sql
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.agents.backends.openai import OpenAIBackend
from app.config import ROOT, Settings
from app.schemas import (
    ChatProfileUpdate,
    FollowupQuestions,
    InfoAnswer,
    InfoDecision,
    Plan,
    ProfileSections,
)
from app.service import ChatService
from app.storage.database import initialize_database, make_engine
from app.storage.dataset_repository import DatasetRepository
from app.storage.models import DatasetConversation, DatasetTurn, Slot
from app.storage.schedule_repository import ScheduleRepository


class EmptyRetriever:
    def search(self, _query, job_id=None):
        return []


@pytest.fixture
def now():
    return datetime(2026, 10, 20, 8, tzinfo=UTC)


@pytest.fixture
def repository():
    repository = ScheduleRepository(make_engine(Settings(_env_file=None).database_url))
    repository.initialize()
    with Session(repository.engine) as session, session.begin():
        for hour in (9, 10, 11):
            session.add(
                Slot(starts_at=datetime(2026, 10, 20, hour), position="Python Dev", available=True)
            )
        session.add(
            Slot(starts_at=datetime(2026, 10, 20, 12), position="Python Dev", available=False)
        )
        session.add(Slot(starts_at=datetime(2026, 10, 20, 13), position="Analyst", available=True))
    yield repository
    repository.engine.dispose()


@pytest.fixture
def calendar_request():
    return {
        "date_text": None,
        "time_text": None,
        "period": None,
        "selected_slot_id": None,
        "accepts_slot": False,
    }


@pytest.fixture
def backend(calendar_request):
    client = Mock(spec=OpenAIBackend)
    client.plan.return_value = Plan(
        next_advisor="respond",
        stop_requested=False,
        closure_reason=None,
        declined_slot_ids=[],
        info_question=None,
        scheduling_needed=False,
        next_question="What Python projects have you worked on?",
    )
    client.exit_label.return_value = "do_not_end"
    client.extract_chat_profile.return_value = ChatProfileUpdate(
        facts=ProfileSections(name=None, experience=[], skills=[], education=[], projects=[]),
        superseded_fact_ids=[],
        confirmed_fact_ids=[],
    )

    def next_plan(state):
        base = client.plan.return_value
        done = {item["advisor"] for item in state.get("advisor_results", [])}
        choice = (
            "exit"
            if "exit" not in done
            else "scheduling"
            if base.scheduling_needed and "scheduling" not in done
            else "info"
            if base.info_question and "info" not in done
            else "respond"
        )
        return base.model_copy(update={"next_advisor": choice})

    client.plan.side_effect = next_plan
    client.schedule.side_effect = lambda state, tool: {
        **tool.invoke(calendar_request),
        "decision": "schedule",
    }
    client.info_decision.side_effect = lambda state: (
        InfoDecision(decision="info_needed", question=state["turn_intent"]["info_question"])
        if state["turn_intent"].get("info_question")
        else InfoDecision(decision="info_not_needed", question=None)
    )
    client.info.return_value = InfoAnswer(
        answer="A recruiter can clarify that.", source_ids=[], supported=False
    )
    client.suggest_followups.return_value = FollowupQuestions(
        questions=[
            "What should I prepare for the interview?",
            "What can you tell me about the team?",
            "Which technologies would I use day to day?",
        ]
    )
    return client


@pytest.fixture
def service(repository, backend):
    service = ChatService(
        Settings(_env_file=None),
        backend=backend,
        repository=repository,
        retriever=EmptyRetriever(),
        checkpointer=InMemorySaver(),
    )
    yield service
    service.close()


@pytest.fixture
def database_factory():
    parsed = make_url(os.environ.get("TEST_DATABASE_URL") or Settings().database_url)
    admin = parsed.set(drivername="postgresql", database="postgres").render_as_string(
        hide_password=False
    )
    names = []
    with psycopg.connect(admin, autocommit=True) as connection:

        def create():
            name = "test_recruitment_" + uuid4().hex
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
            names.append(name)
            return parsed.set(database=name).render_as_string(hide_password=False)

        try:
            yield create
        finally:
            for name in reversed(names):
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
                )


@pytest.fixture(autouse=True)
def isolated_storage(request, monkeypatch, tmp_path):
    # Pure input/dataset checks do not need a running database or copied PDFs.
    if request.node.path.parent.name != "integration":
        return
    database_factory = request.getfixturevalue("database_factory")
    monkeypatch.setenv("DATABASE_URL", database_factory())
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    backend = request.getfixturevalue("backend")
    monkeypatch.setattr("app.service.OpenAIBackend", lambda settings: backend)
    embeddings = Mock(spec=["embed_documents", "embed_query"])
    vector = [1.0] + [0.0] * 511
    embeddings.embed_documents.side_effect = lambda texts: [list(vector) for _ in texts]
    embeddings.embed_query.return_value = vector
    monkeypatch.setattr("app.rag.retriever.OpenAIEmbeddings", lambda **kwargs: embeddings)

    def reject_live_http(*args, **kwargs):
        raise AssertionError("Use an HTTP mock; tests must never call live APIs")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_live_http)
    # HTTP mocks use reserved example domains; avoid real DNS during logo tests.
    monkeypatch.setattr("app.ingestion.logos.public_addresses", lambda host: {"93.184.216.34"})
    monkeypatch.setenv("JOB_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("JOB_MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("INGESTION_REPORT_PATH", str(tmp_path / "reports"))
    monkeypatch.setenv("CONVERSATION_EXPORT_PATH", str(tmp_path / "conversations.json"))
    monkeypatch.setenv("JOB_SOURCES_PATH", str(tmp_path / "sources.json"))
    directory = tmp_path / "data/jobs"
    directory.mkdir(parents=True)
    original = ROOT / "data/jobs/Python Developer Job Description.pdf"
    shutil.copyfile(original, directory / original.name)


@pytest.fixture
def benchmark_conversations():
    # Small test-only records keep tests independent of private benchmark backups.
    records = []
    for cid, split in enumerate(("train", "validation", "test"), 1):
        start = datetime(2026, 10, cid, 8, tzinfo=UTC)
        turns = [
            ("recruiter", "What is your Python experience?", "continue"),
            ("candidate", "I use Python.", None),
            ("recruiter", "Choose an appointment.", "schedule"),
            ("candidate", "I accept.", None),
            ("recruiter", "Thank you. Goodbye.", "end"),
        ]
        records.append(
            {
                "conversation_id": cid,
                "dataset_version": "test-v1",
                "split": split,
                "start_time_utc": start.isoformat().replace("+00:00", "Z"),
                "candidate_phone": None,
                "recruiter_phone": None,
                "turns": [
                    {
                        "turn_id": index,
                        "speaker": speaker,
                        "timestamp_utc": (start + timedelta(minutes=index - 1))
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "text": text,
                        "label": label,
                    }
                    for index, (speaker, text, label) in enumerate(turns, 1)
                ],
            }
        )
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return [{**record, "source_sha256": fingerprint} for record in records]


@pytest.fixture
def benchmark_dataset(monkeypatch, benchmark_conversations):
    monkeypatch.setenv("DATASET_VERSION", "test-v1")
    settings = Settings()
    engine = make_engine(settings.database_url)
    initialize_database(engine)
    with Session(engine) as session, session.begin():
        session.add_all(
            [
                DatasetConversation(
                    dataset_version=row["dataset_version"],
                    conversation_id=row["conversation_id"],
                    started_at=datetime.fromisoformat(row["start_time_utc"].replace("Z", "+00:00")),
                    split=row["split"],
                    source_sha256=row["source_sha256"],
                    source_metadata={"candidate_phone": None, "recruiter_phone": None},
                )
                for row in benchmark_conversations
            ]
        )
        session.flush()
        session.add_all(
            [
                DatasetTurn(
                    dataset_version=row["dataset_version"],
                    conversation_id=row["conversation_id"],
                    turn_id=turn["turn_id"],
                    speaker=turn["speaker"],
                    occurred_at=datetime.fromisoformat(
                        turn["timestamp_utc"].replace("Z", "+00:00")
                    ),
                    text=turn["text"],
                    expected_action=turn["label"],
                )
                for row in benchmark_conversations
                for turn in row["turns"]
            ]
        )
    try:
        yield DatasetRepository(engine, settings.dataset_version)
    finally:
        engine.dispose()
