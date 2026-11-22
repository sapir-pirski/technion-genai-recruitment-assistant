"""Controlled multi-turn replay; SQL offers and bookings persist within each case."""

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import AwareDatetime, Field, ValidationError, model_validator
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateSchema, DropSchema

from app.agents.backends.common import prompt_manifest
from app.graph.builder import build_graph
from app.ingestion.models import DEFAULT_JOB_ID
from app.schemas import Action, StrictModel, TurnResult
from app.service import ChatService
from app.storage.database import make_engine
from app.storage.models import Booking, Slot
from app.storage.schedule_repository import ScheduleRepository, utc_naive


class ExpectedState(StrictModel):
    action: Action
    closed: bool
    bookings: int = Field(ge=0, le=1)
    minimum_offers: int = Field(default=0, ge=0, le=3)
    maximum_offers: int = Field(default=3, ge=0, le=3)
    alternative_count: int | None = Field(default=None, ge=0, le=3)
    preferred_date: date | None = None
    only_offer_date: date | None = None
    preference_origin_time: AwareDatetime | None = None
    cited_answer: bool = False
    closure_reason: Literal["opt_out", "scheduled"] | None = None


class ReplayStep(StrictModel):
    text: str = Field(min_length=1, max_length=4000)
    elapsed_seconds: int | None = Field(default=None, ge=0, le=2678400)
    selected_ordinal: int | None = Field(default=None, ge=1, le=3)
    expect: ExpectedState


class ReplayCase(StrictModel):
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    reference_time: AwareDatetime
    timezone: str
    calendar_starts: list[AwareDatetime] = Field(max_length=50)
    job_description: str = Field(min_length=1, max_length=6000)
    steps: list[ReplayStep] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def valid_calendar(self):
        ZoneInfo(self.timezone)
        starts = sorted(self.calendar_starts)
        offsets = [
            step.elapsed_seconds if step.elapsed_seconds is not None else i
            for i, step in enumerate(self.steps)
        ]
        if offsets != sorted(offsets):
            raise ValueError("Replay turns must be in timestamp order")
        if starts and starts[0] <= self.reference_time + timedelta(seconds=max(offsets)):
            raise ValueError("Replay appointments must be after every replay turn")
        if any(b < a + timedelta(minutes=30) for a, b in zip(starts, starts[1:], strict=False)):
            raise ValueError("Replay appointments must not overlap")
        return self


def load_replays(path):
    if path.stat().st_size > 1_000_000:
        raise ValueError("Workflow input must be at most 1 MB")
    try:
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not 1 <= len(rows) <= 50:
            raise ValueError
        cases = [ReplayCase.model_validate(row) for row in rows]
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError
        return cases
    except (ValidationError, ValueError, TypeError, KeyError):
        raise ValueError(
            "Invalid workflow cases; check IDs, timestamps, calendar and step fields"
        ) from None


class CaseRetriever:
    def __init__(self, text):
        self.text = text

    def search(self, question, job_id=None):
        return [{"id": "replay-source", "text": self.text, "job_id": DEFAULT_JOB_ID}]


def evaluate_workflows(settings, source: Path, *, output=None, backend=None):
    """Use explicit fictitious cases, never the private benchmark or production calendar."""
    cases = load_replays(source)
    settings.require_api_key()
    engine = make_engine(settings.database_url)
    schema = "evaluation_workflow_" + uuid4().hex
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    rows = []
    versions = prompt_manifest(settings)
    try:
        repository = ScheduleRepository(
            engine.execution_options(schema_translate_map={None: schema})
        )
        service = ChatService(
            settings,
            backend=backend,
            repository=repository,
            retriever=CaseRetriever(""),
            checkpointer=InMemorySaver(),
        )
        try:
            if isinstance(getattr(service.backend, "prompt_manifest", None), dict):
                versions = service.backend.prompt_manifest
            for case in cases:
                # Deletes are restricted to this disposable calendar schema.
                with Session(repository.engine) as session, session.begin():
                    session.execute(delete(Booking))
                    session.execute(delete(Slot))
                    session.add_all(
                        [
                            Slot(
                                starts_at=utc_naive(start),
                                job_id=DEFAULT_JOB_ID,
                                available=True,
                                duration_minutes=30,
                            )
                            for start in case.calendar_starts
                        ]
                    )
                graph = build_graph(
                    service.backend,
                    repository,
                    CaseRetriever(case.job_description),
                    InMemorySaver(),
                    followups=False,
                )
                thread = str(uuid4())
                config = service._config(thread)
                state = {}
                with service.traces.trace(
                    None,
                    str(uuid4()),
                    component="evaluation",
                    operation="workflow",
                    attributes={"case_id": case.case_id, "controlled_calendar": True},
                ) as trace:
                    for index, step in enumerate(case.steps):
                        error = None
                        checks = {}
                        actual = {}
                        try:
                            if state.get("conversation_status") == "closed":
                                raise ValueError("Conversation closed before remaining steps")
                            offered = state.get("offered_slots", [])
                            selected = (
                                offered[step.selected_ordinal - 1]["id"]
                                if step.selected_ordinal
                                else None
                            )
                            messages = [
                                *state.get("messages", []),
                                {"role": "user", "content": step.text},
                            ]
                            payload = service._turn_payload(
                                state,
                                thread,
                                str(uuid4()),
                                DEFAULT_JOB_ID,
                                {"title": "Python Developer", "company_name": "Example Company"},
                                {"scheduling_enabled": True},
                                messages,
                                case.reference_time
                                + timedelta(
                                    seconds=step.elapsed_seconds
                                    if step.elapsed_seconds is not None
                                    else index
                                ),
                                case.timezone,
                                selected,
                                [],
                            )
                            # Expected outcomes and selector ordinals never enter model context.
                            state = graph.invoke(payload, config, durability="sync")
                            result = TurnResult.model_validate(state["result"])
                            receipt = repository.get_booking(thread, case.timezone, DEFAULT_JOB_ID)
                            with Session(repository.engine) as session:
                                count = session.scalar(
                                    select(func.count())
                                    .select_from(Booking)
                                    .where(Booking.conversation_id == thread)
                                )
                            actual = {
                                "action": result.action,
                                "closed": result.conversation_status == "closed",
                                "bookings": count,
                                "offers": len(result.offered_slots),
                                "closure_reason": result.closure_reason,
                                "cited_answer": bool(result.source_ids),
                                "alternative_count": sum(
                                    bool(slot.get("is_alternative"))
                                    for slot in result.offered_slots
                                ),
                                "preferred_date": state.get("schedule_preferences", {}).get("date"),
                                "preference_origin_time": state.get("schedule_preferences", {}).get(
                                    "date_reference_time"
                                ),
                            }
                            checks = {
                                "action": actual["action"] == step.expect.action,
                                "closed": actual["closed"] == step.expect.closed,
                                "bookings": count == step.expect.bookings,
                                "offers": actual["offers"] >= step.expect.minimum_offers,
                                "maximum_offers": actual["offers"] <= step.expect.maximum_offers,
                                "alternative_count": step.expect.alternative_count is None
                                or actual["alternative_count"] == step.expect.alternative_count,
                                "preferred_date": actual["preferred_date"]
                                == (
                                    step.expect.preferred_date.isoformat()
                                    if step.expect.preferred_date
                                    else None
                                ),
                                "preference_origin_time": step.expect.preference_origin_time is None
                                or actual["preference_origin_time"]
                                == step.expect.preference_origin_time.isoformat(),
                                "required_date": step.expect.only_offer_date is None
                                or all(
                                    slot["local_time"].startswith(
                                        step.expect.only_offer_date.isoformat()
                                    )
                                    for slot in result.offered_slots
                                ),
                                "citations": not step.expect.cited_answer or actual["cited_answer"],
                                "closure_reason": result.closure_reason
                                == step.expect.closure_reason,
                                "receipt": result.booking == receipt,
                                "scheduled_closure_has_receipt": result.closure_reason
                                != "scheduled"
                                or receipt is not None,
                            }
                            error = result.error
                        except Exception as exc:
                            error = type(exc).__name__
                        passed = bool(checks) and all(checks.values()) and not error
                        if not passed:
                            trace.fail(error or "WorkflowExpectationFailed")
                        rows.append(
                            {
                                "case_id": case.case_id,
                                "step": index + 1,
                                "passed": passed,
                                "checks": checks,
                                "actual": actual,
                                "error": error,
                            }
                        )
        finally:
            service.close()
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()
    report = {
        "provider": "openai",
        "task": "workflow",
        "evaluation_mode": "controlled_stateful_replay",
        "created_at": datetime.now(UTC).isoformat(),
        "model_policy": settings.model_policy(),
        "prompt_manifest": versions,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "summary": {
            "cases": len(cases),
            "steps": len(rows),
            "passed_steps": sum(row["passed"] for row in rows),
            "passed_cases": sum(
                all(row["passed"] for row in rows if row["case_id"] == case.case_id)
                for case in cases
            ),
            "execution_failures": sum(bool(row["error"]) for row in rows),
        },
        "rows": rows,
        "limitations": [
            "Fictitious controlled cases; not historical benchmark accuracy.",
            "Follow-up generation and uploaded profile processing are outside this booking replay.",
        ],
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        output.chmod(0o600)
    return report
