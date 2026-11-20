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
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.agents.backends.openai import OpenAIBackend
from app.config import ROOT, Settings
from app.rag.repository import JobDescriptionRepository
from app.scheduling.models import Slot
from app.scheduling.repository import ScheduleRepository
from app.schemas import (
    InfoAnswer,
    InfoDecision,
    Plan,
)
from app.storage.database import make_engine
from app.workflow.service import ChatService


class EmptyRetriever:
    def search(self, _query, job_id=None):
        return []


@pytest.fixture
def legacy_job():
    """Simulate an existing imported posting without restoring collection features."""
    registry = JobDescriptionRepository(Settings(_env_file=None))
    try:
        with registry.engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE public.jobs (job_id text PRIMARY KEY, status text, description_text text)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO public.jobs VALUES ('reference', 'active', 'Legacy-only salary USD 200000.')"
                )
            )
        return "reference"
    finally:
        registry.close()


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

    def create():
        name = "test_recruitment_" + uuid4().hex
        with psycopg.connect(admin, autocommit=True, connect_timeout=10) as connection:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
        names.append(name)
        return parsed.set(database=name).render_as_string(hide_password=False)

    try:
        yield create
    finally:
        # Reconnect for cleanup instead of keeping an idle hosted connection per test.
        for name in reversed(names):
            with psycopg.connect(admin, autocommit=True, connect_timeout=10) as connection:
                connection.execute(
                    sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name))
                )


@pytest.fixture(autouse=True)
def isolated_storage(request, monkeypatch, tmp_path):
    # Behavior tests use explicit mock baselines; model artifacts are tested separately.
    monkeypatch.setenv("EXIT_PROVIDER", "openai")
    # Pure input/dataset checks do not need a running database or copied PDFs.
    if request.node.path.parent.name != "integration":
        return
    database_factory = request.getfixturevalue("database_factory")
    monkeypatch.setenv("DATABASE_URL", database_factory())
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    backend = request.getfixturevalue("backend")
    monkeypatch.setattr("app.workflow.service.OpenAIBackend", lambda settings: backend)
    monkeypatch.setattr("app.evaluation.runner.OpenAIBackend", lambda settings: backend)
    embeddings = Mock(spec=["embed_documents", "embed_query"])
    vector = [1.0] + [0.0] * 511
    embeddings.embed_documents.side_effect = lambda texts: [list(vector) for _ in texts]
    embeddings.embed_query.return_value = vector
    monkeypatch.setattr("app.rag.retriever.OpenAIEmbeddings", lambda **kwargs: embeddings)

    def reject_live_http(*args, **kwargs):
        raise AssertionError("Use an HTTP mock; tests must never call live APIs")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", reject_live_http)
    monkeypatch.setenv("JOB_DATA_PATH", str(tmp_path / "data"))
    monkeypatch.setenv("JOB_MANIFEST_PATH", str(tmp_path / "manifest.json"))
    monkeypatch.setenv("CONVERSATION_EXPORT_PATH", str(tmp_path / "conversations.json"))
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
def benchmark_dataset(monkeypatch, benchmark_conversations, tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(benchmark_conversations, indent=2))
    path.chmod(0o600)
    monkeypatch.setenv("DATASET_VERSION", "test-v1")
    monkeypatch.setenv("DATASET_PATH", str(path))
    return path
