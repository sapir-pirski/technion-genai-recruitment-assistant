"""Offline answer grading from explicitly supplied, self-contained evidence packets."""

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from uuid import uuid4

from pydantic import Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from app.observability.logging import log_event
from app.observability.tracing import TraceRecorder
from app.schemas import JudgeRating, JudgeResult, QualityScore, StrictModel
from app.storage.database import initialize_database, make_engine
from app.storage.models import TraceSpan

RUBRIC_VERSION = "quality-v1"
DIMENSIONS = ("grounding", "completeness", "relevance_clarity", "followup_quality")
CONTEXT_REFS = {"question", "history", "answer", "followups"}


def digest(value):
    text = (
        value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    )
    return hashlib.sha256(text.encode()).hexdigest()


class QualityMessage(StrictModel):
    role: str
    content: str = Field(min_length=1, max_length=2000)

    @field_validator("role")
    @classmethod
    def dialogue_role(cls, value):
        if value not in {"user", "assistant"}:
            raise ValueError("History must contain only user and assistant messages")
        return value


class QualityEvidence(StrictModel):
    evidence_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    text: str = Field(min_length=1, max_length=6000)

    @model_validator(mode="after")
    def usable_evidence(self):
        if self.evidence_id.casefold() in CONTEXT_REFS or not self.text.strip():
            raise ValueError("Evidence needs a distinct ID and nonempty text")
        return self


class HumanQualityScores(StrictModel):
    grounding: QualityScore | None = None
    completeness: QualityScore | None = None
    relevance_clarity: QualityScore | None = None
    followup_quality: QualityScore | None = None

    @field_validator(*DIMENSIONS, mode="before")
    @classmethod
    def integer_scores(cls, value):
        if value is not None and type(value) is not int:
            raise ValueError("Human reference scores must be integers or null")
        return value


class QualityCase(StrictModel):
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    question: str = Field(min_length=1, max_length=4000)
    history: list[QualityMessage] = Field(default_factory=list, max_length=12)
    answer: str = Field(min_length=1, max_length=6000)
    followups: list[str] = Field(default_factory=list, max_length=10)
    followups_expected: bool | None = Field(default=None, strict=True)
    sources: list[QualityEvidence] = Field(default_factory=list, max_length=20)
    calendar_receipts: list[QualityEvidence] = Field(default_factory=list, max_length=10)
    evidence_complete: bool = Field(default=False, strict=True)
    human_scores: HumanQualityScores | None = None

    def context(self):
        # Deliberate allowlist: never send case identity, human scores or model provenance.
        return self.model_dump(
            include={
                "question",
                "history",
                "answer",
                "followups",
                "followups_expected",
                "sources",
                "calendar_receipts",
                "evidence_complete",
            }
        )

    @model_validator(mode="after")
    def bounded_packet(self):
        ids = [item.evidence_id for item in self.sources + self.calendar_receipts]
        if len(ids) != len(set(ids)):
            raise ValueError("Evidence IDs must be unique across sources and receipts")
        if not self.question.strip() or not self.answer.strip():
            raise ValueError("A question and answer are required")
        if any(not value.strip() or len(value) > 1000 for value in self.followups):
            raise ValueError("Follow-ups must be nonempty and bounded")
        if len(json.dumps(self.context(), ensure_ascii=False)) > 50000:
            raise ValueError("Quality evidence packet is too large")
        return self


def load_cases(path: Path, limit=20):
    if not 1 <= limit <= 200:
        raise ValueError("Use a quality limit between 1 and 200")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Quality case files must be at most 2 MiB")
    try:
        raw = path.read_text()
        values = json.loads(raw)
        if not isinstance(values, list) or not 1 <= len(values) <= 200:
            raise ValueError
        cases = [QualityCase.model_validate(value) for value in values]
        if len({case.case_id for case in cases}) != len(cases):
            raise ValueError
    except (ValueError, TypeError):
        # Pydantic errors can contain source text; do not expose them through CLI stderr.
        raise ValueError(
            "Quality input must be a valid list of distinct, bounded evidence packets"
        ) from None
    return cases[:limit], len(cases), digest(raw)


def assess_case(backend, case: QualityCase) -> JudgeResult:
    result = JudgeResult.model_validate(backend.judge_quality(case.context()))
    evidence_ids = {item.evidence_id for item in case.sources + case.calendar_receipts}
    allowed = evidence_ids | CONTEXT_REFS
    references = [ref for name in DIMENSIONS for ref in getattr(result, name).evidence_refs]
    references += [ref for issue in result.issues for ref in issue.evidence_refs]
    if not set(references).issubset(allowed):
        raise ValueError("Judge cited evidence outside the supplied packet")
    updates = {}
    unavailable = set()
    if not case.evidence_complete or not evidence_ids:
        unavailable.add("grounding")
        updates["grounding"] = JudgeRating(
            status="not_assessable",
            score=None,
            reason="Complete source or receipt evidence was not supplied.",
            evidence_refs=[],
        )
    elif result.grounding.status == "scored" and not evidence_ids.intersection(
        result.grounding.evidence_refs
    ):
        raise ValueError("A grounding score must refer to supplied source or receipt evidence")
    if case.followups_expected is not True:
        unavailable.add("followup_quality")
        updates["followup_quality"] = JudgeRating(
            status="not_assessable",
            score=None,
            reason="Follow-ups were not required, or their applicability was not supplied.",
            evidence_refs=[],
        )
    elif not case.followups:
        updates["followup_quality"] = JudgeRating(
            status="scored",
            score=1,
            reason="Follow-ups were required but none were supplied.",
            evidence_refs=["followups"],
        )
    updates["issues"] = [
        issue
        for issue in result.issues
        if issue.dimension not in unavailable
        and getattr(result, issue.dimension).score not in {None, 5}
    ]
    return result.model_copy(update=updates)


def summarize_rows(rows):
    summary = {"cases": len(rows), "completed": sum(row["status"] == "completed" for row in rows)}
    summary["failed"] = summary["cases"] - summary["completed"]
    dimensions = {}
    comparisons = []
    reference_count = 0
    for name in DIMENSIONS:
        ratings = [row["assessment"][name] for row in rows if row["assessment"] is not None]
        scores = [rating["score"] for rating in ratings if rating["score"] is not None]
        dimensions[name] = {
            "scored": len(scores),
            "not_assessable": len(ratings) - len(scores),
            "failed": summary["failed"],
            "mean": round(mean(scores), 4) if scores else None,
            "distribution": {str(score): scores.count(score) for score in range(1, 6)},
        }
        for row in rows:
            human = (row["human_scores"] or {}).get(name)
            if human is not None:
                reference_count += 1
                predicted = row["assessment"][name]["score"] if row["assessment"] else None
                if predicted is not None:
                    comparisons.append(abs(predicted - human))
    return {
        **summary,
        "dimensions": dimensions,
        "human_reference_agreement": {
            "provided_ratings": reference_count,
            "comparable_ratings": len(comparisons),
            "unscored_reference_ratings": reference_count - len(comparisons),
            "mean_absolute_error": round(mean(comparisons), 4) if comparisons else None,
            "exact_agreement": comparisons.count(0) / len(comparisons) if comparisons else None,
            "within_one_agreement": sum(value <= 1 for value in comparisons) / len(comparisons)
            if comparisons
            else None,
        },
    }


def trace_usage(engine, request_id):
    try:
        with Session(engine) as session:
            spans = list(
                session.scalars(select(TraceSpan).where(TraceSpan.request_id == request_id))
            )
            root = next((span for span in spans if span.parent_span_id is None), None)
            calls = [
                span for span in spans if span.component == "llm" and span.operation == "judge"
            ]
            costs = [span.attributes.get("estimated_cost_usd") for span in calls]
            return {
                "trace_id": root.trace_id if root else None,
                "recorded_calls": len(calls),
                "missing_trace": not calls,
                "model_duration_ms": round(sum(span.duration_ms for span in calls), 3)
                if calls
                else None,
                "input_tokens": sum(span.input_tokens or 0 for span in calls)
                if calls and all(span.input_tokens is not None for span in calls)
                else None,
                "output_tokens": sum(span.output_tokens or 0 for span in calls)
                if calls and all(span.output_tokens is not None for span in calls)
                else None,
                "cached_input_tokens": sum(span.cached_input_tokens or 0 for span in calls)
                if calls and all(span.cached_input_tokens is not None for span in calls)
                else None,
                "estimated_cost_usd": round(sum(value for value in costs if value is not None), 10)
                if any(value is not None for value in costs)
                else None,
                "unknown_cost_calls": sum(value is None for value in costs),
                "models": sorted({span.model for span in calls if span.model}),
            }
    except Exception as error:
        log_event(
            "quality_trace_read_failed", component="app.evaluation", error_code=type(error).__name__
        )
        return {
            "trace_id": None,
            "recorded_calls": 0,
            "missing_trace": True,
            "model_duration_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "cached_input_tokens": None,
            "estimated_cost_usd": None,
            "unknown_cost_calls": 0,
            "models": [],
        }


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".quality-",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.fchmod(handle.fileno(), 0o600)
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def evaluate_quality(settings: Settings, input_path: Path, *, output: Path, limit=20, backend=None):
    if input_path.resolve() == output.resolve() or output.suffix.lower() != ".json":
        raise ValueError("Use a distinct JSON output path for the quality report")
    cases, available, input_hash = load_cases(input_path, limit)
    settings.require_api_key()
    backend = backend or OpenAIBackend(settings)
    engine = make_engine(settings.database_url)
    rows = []
    try:
        initialize_database(engine)
        recorder = TraceRecorder(engine)
        for index, case in enumerate(cases):
            request_id = str(uuid4())
            result, error = None, None
            try:
                with recorder.trace(
                    None,
                    request_id,
                    component="evaluation",
                    operation="quality",
                    attributes={
                        "case_index": index,
                        "case_sha256": digest(case.context()),
                        "rubric_version": RUBRIC_VERSION,
                    },
                ) as root:
                    result = assess_case(backend, case)
                    root.attributes["scored_dimensions"] = sum(
                        getattr(result, name).score is not None for name in DIMENSIONS
                    )
            except Exception as failure:
                error = type(failure).__name__
            rows.append(
                {
                    "case_id": case.case_id,
                    "input_sha256": digest(case.context()),
                    "status": "failed" if error else "completed",
                    "error_code": error,
                    "assessment": result.model_dump() if result else None,
                    "human_scores": case.human_scores.model_dump() if case.human_scores else None,
                    "evidence": {
                        item.evidence_id: digest(item.text)
                        for item in case.sources + case.calendar_receipts
                    },
                    "usage": trace_usage(engine, request_id),
                }
            )
    finally:
        engine.dispose()
    costs = [row["usage"]["estimated_cost_usd"] for row in rows]
    report = {
        "task": "quality",
        "rubric_version": RUBRIC_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "input_sha256": input_hash,
        "available_cases": available,
        "selected_cases": len(cases),
        "judge_model": settings.judge_model,
        "judge_policy": settings.model_policy()["judge"],
        "judge_prompt": backend.prompt_manifest["judge"],
        "summary": summarize_rows(rows),
        "usage": {
            "recorded_calls": sum(row["usage"]["recorded_calls"] for row in rows),
            "estimated_cost_usd": round(sum(value for value in costs if value is not None), 10)
            if any(value is not None for value in costs)
            else None,
            "unknown_cost_calls": sum(row["usage"]["unknown_cost_calls"] for row in rows),
            "missing_trace_cases": sum(row["usage"]["missing_trace"] for row in rows),
            "reported_input_tokens": sum(row["usage"]["input_tokens"] or 0 for row in rows),
            "reported_output_tokens": sum(row["usage"]["output_tokens"] or 0 for row in rows),
            "unreported_input_cases": sum(row["usage"]["input_tokens"] is None for row in rows),
            "unreported_output_cases": sum(row["usage"]["output_tokens"] is None for row in rows),
            "mean_model_duration_ms": round(
                mean(
                    row["usage"]["model_duration_ms"]
                    for row in rows
                    if row["usage"]["model_duration_ms"] is not None
                ),
                3,
            )
            if any(row["usage"]["model_duration_ms"] is not None for row in rows)
            else None,
        },
        "rows": rows,
        "limitations": [
            "Judge scores are advisory estimates, not verified labels or database validation.",
            "The default judge is also used to generate some answers; calibrate against human ratings and consider shared-model bias.",
            "Human reference scores are caller supplied, not independently verified, and are never sent to the judge.",
            "Evidence is supplied as-is; the evaluator does not fetch current PDFs or certify receipt authenticity.",
            "Reported Standard-tier cost excludes unreported usage and SDK-internal retries.",
            "Case-set means describe the supplied answers, including intentional failures, not production chatbot quality.",
        ],
    }
    write_report(output, report)
    return report
