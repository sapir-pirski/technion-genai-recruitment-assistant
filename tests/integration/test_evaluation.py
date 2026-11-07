import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from langchain_openai import ChatOpenAI
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.backends.common import prompt_manifest, prompt_spec
from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from app.evaluation.dataset import SPLIT_NAMES, load_examples, prepare_dataset, split_examples
from app.evaluation.judge import evaluate_quality
from app.evaluation.runner import evaluate
from app.evaluation.workflow import evaluate_workflows
from app.fine_tuning.jobs import validate_jsonl
from app.storage.dataset_repository import DatasetRepository
from app.storage.models import Booking, Conversation, ConversationMessage, DatasetTurn, TraceSpan


def test_postgres_dataset_roundtrip_and_training_export(
    benchmark_dataset, benchmark_conversations, tmp_path
):
    assert benchmark_dataset.load_conversations() == benchmark_conversations
    settings = Settings()
    examples = load_examples(settings)
    assert len(examples) == 9
    assert Counter(row["label"] for row in examples) == {"continue": 3, "schedule": 3, "end": 3}
    manifest = prepare_dataset(settings, tmp_path)
    assert manifest["source"] == "postgresql" and manifest["dataset_version"] == "test-v1"
    assert manifest["splits"] == {"train": [1], "validation": [2], "test": [3]}
    assert manifest["source_sha256"] == benchmark_conversations[0]["source_sha256"]
    assert len(manifest["dataset_sha256"]) == 64
    assert manifest["exit_training"]["base_model"] == settings.chat_model
    assert manifest["exit_training"]["instruction_role"] == "system"
    assert (
        manifest["exit_training"]["prompt_sha256"]
        == prompt_spec("exit", settings.chat_model)["prompt_sha256"]
    )
    for split in SPLIT_NAMES:
        assert validate_jsonl(tmp_path / f"exit_{split}.jsonl") == 3
        training_rows = [
            json.loads(line) for line in (tmp_path / f"exit_{split}.jsonl").read_text().splitlines()
        ]
        spec = prompt_spec("exit", settings.chat_model)
        assert all(
            row["messages"][0] == {"role": spec["instruction_role"], "content": spec["content"]}
            for row in training_rows
        )
        rows = json.loads((tmp_path / f"{split}.json").read_text())
        assert rows == split_examples(examples, split)
        assert {row["conversation_id"] for row in rows} == set(manifest["splits"][split])
    with pytest.raises(ValueError, match="Restore dataset_conversations"):
        DatasetRepository(benchmark_dataset.engine, "missing-version").load_conversations()


@pytest.mark.parametrize("task", ["actions", "exit"])
def test_evaluation_uses_stored_splits_and_preserves_benchmark(benchmark_dataset, task, backend):
    before = benchmark_dataset.load_conversations()
    report = evaluate(Settings(), split="test", task=task)
    assert report["all_turns"]["count"] == 3
    assert report["dataset_version"] == "test-v1" and len(report["dataset_sha256"]) == 64
    assert {row["conversation_id"] for row in report["rows"]} == {3}
    assert report["prompt_manifest"] == prompt_manifest(Settings())
    assert (
        report["evaluation_mode"] == "historical_prefix" and report["booking_correctness"] is None
    )
    assert benchmark_dataset.load_conversations() == before
    with Session(benchmark_dataset.engine) as session:
        assert session.scalar(select(func.count()).select_from(Conversation)) == 0
    if task == "exit":
        backend.exit_label.side_effect = TimeoutError("private provider error")
    else:
        backend.plan.side_effect = TimeoutError("private provider error")
    failed = evaluate(Settings(), split="test", task=task)
    assert failed["all_turns"]["failures"] > 0
    assert all(row["predicted"] is None for row in failed["rows"] if row["error"])
    assert "private provider error" not in json.dumps(failed)


def test_stateful_workflow_replay_preserves_public_calendar_and_excludes_expectations(
    repository, backend, tmp_path
):
    from app.schemas import Plan
    from app.storage.models import Slot

    with Session(repository.engine) as session:
        before = session.scalar(select(func.count()).select_from(Slot))
    cases = json.loads(Path("tests/fixtures/workflow_cases.json").read_text())
    case = cases[1]
    case["steps"].pop(1)  # Keep an offer followed by a selected, explicitly confirmed slot.
    source = tmp_path / "workflow.json"
    source.write_text(json.dumps([case]))
    seen = []

    def plan(state):
        seen.append(state)
        done = {item["advisor"] for item in state.get("advisor_results", [])}
        return Plan(
            next_advisor="respond" if "scheduling" in done else "scheduling",
            stop_requested=False,
            closure_reason=None,
            scheduling_needed=True,
            info_question=None,
            declined_slot_ids=[],
            next_question="Which appointment works?",
        )

    backend.plan.side_effect = plan
    output = tmp_path / "report.json"
    report = evaluate_workflows(Settings(), source, output=output, backend=backend)
    assert report["summary"]["passed_steps"] == 2 and report["summary"]["passed_cases"] == 1
    assert report["rows"][-1]["actual"]["closure_reason"] == "scheduled"
    assert all(
        "expect" not in json.dumps(state) and "selected_ordinal" not in json.dumps(state)
        for state in seen
    )
    assert "Please confirm" not in output.read_text() and output.stat().st_mode & 0o777 == 0o600
    with Session(repository.engine) as session:
        assert session.scalar(select(func.count()).select_from(Slot)) == before
        assert session.scalar(select(func.count()).select_from(Booking)) == 0
        assert session.scalar(select(func.count()).select_from(Conversation)) == 0


def test_database_rejects_candidate_gold_label(benchmark_dataset):
    before = benchmark_dataset.load_conversations()
    with (
        pytest.raises(IntegrityError),
        Session(benchmark_dataset.engine) as session,
        session.begin(),
    ):
        session.add(
            DatasetTurn(
                dataset_version="test-v1",
                conversation_id=1,
                turn_id=99,
                speaker="candidate",
                occurred_at=datetime.now(UTC),
                text="Invalid test label",
                expected_action="end",
            )
        )
        session.flush()
    assert benchmark_dataset.load_conversations() == before


def packet(case_id="test-answer"):
    return {
        "case_id": case_id,
        "question": "Which skills are required?",
        "answer": "Private answer text: Python.",
        "sources": [
            {"evidence_id": "source-1", "text": "Private source text: Python is required."}
        ],
        "evidence_complete": True,
        "followups_expected": False,
        "human_scores": {"grounding": 4},
    }


def judgment():
    def rating(refs):
        return {
            "status": "scored",
            "score": 5,
            "reason": "Supported by the provided evidence.",
            "evidence_refs": refs,
        }

    return {
        "grounding": rating(["answer", "source-1"]),
        "completeness": rating(["answer", "question"]),
        "relevance_clarity": rating(["answer"]),
        "followup_quality": {
            "status": "not_assessable",
            "score": None,
            "reason": "No follow-ups required.",
            "evidence_refs": [],
        },
        "issues": [],
    }


def test_quality_http_contract_report_trace_and_no_conversation_writes(tmp_path, capfd):
    settings = Settings()
    source, destination = tmp_path / "cases.json", tmp_path / "quality.json"
    source.write_text(json.dumps([packet()]))
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-quality",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini-2024-07-18",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(judgment())},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 220,
                    "total_tokens": 320,
                    "prompt_tokens_details": {"cached_tokens": 20},
                },
            },
        )

    backend = OpenAIBackend(settings)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        backend.judge_client = ChatOpenAI(
            model=settings.judge_model, api_key="test-only-key", http_client=client
        )
        report = evaluate_quality(settings, source, output=destination, backend=backend)
    assert len(requests) == 1 and requests[0]["model"] == settings.judge_model
    assert requests[0]["messages"][0]["role"] == "system" and "tools" not in requests[0]
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["JudgeIssue"]["properties"]["evidence_refs"]["minItems"] == 1
    for name in ("JudgeRating", "JudgeIssue"):
        assert schema["$defs"][name]["properties"]["evidence_refs"]["items"]["enum"] == [
            "question",
            "history",
            "answer",
            "followups",
            "source-1",
        ]
    context = json.loads(requests[0]["messages"][1]["content"])
    assert "human_scores" not in context and "case_id" not in context
    assert report["summary"]["completed"] == 1 and report["summary"]["failed"] == 0
    assert report["summary"]["human_reference_agreement"]["mean_absolute_error"] == 1
    assert report["rows"][0]["usage"]["models"] == ["gpt-4o-mini-2024-07-18"]
    assert report["usage"]["recorded_calls"] == 1 and report["usage"]["estimated_cost_usd"] > 0
    assert destination.stat().st_mode & 0o777 == 0o600
    assert json.loads(destination.read_text()) == report
    assert (
        "Private answer text" not in destination.read_text()
        and "Private source text" not in destination.read_text()
    )
    from app.storage.database import make_engine

    engine = make_engine(settings.database_url)
    try:
        with Session(engine) as session:
            spans = list(session.scalars(select(TraceSpan)))
            model = next(row for row in spans if row.component == "llm")
            assert model.attributes["prompt_sha256"] == report["judge_prompt"]["prompt_sha256"]
            assert (model.input_tokens, model.output_tokens) == (100, 220)
            assert all(row.conversation_id is None for row in spans)
            assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 0
            assert session.scalar(select(func.count()).select_from(Booking)) == 0
            assert "Private" not in json.dumps([row.attributes for row in spans])
    finally:
        engine.dispose()
    assert "Private" not in capfd.readouterr().err


def test_failed_judge_stays_unscored_and_next_case_continues(tmp_path):
    settings = Settings()
    source = tmp_path / "cases.json"
    source.write_text(json.dumps([packet("one"), packet("two")]))
    backend = Mock(
        judge_quality=Mock(side_effect=[TimeoutError("private provider error"), judgment()])
    )
    backend.prompt_manifest = prompt_manifest(settings)
    report = evaluate_quality(settings, source, output=tmp_path / "report.json", backend=backend)
    assert backend.judge_quality.call_count == 2
    assert report["summary"]["failed"] == 1 and report["summary"]["completed"] == 1
    assert (
        report["rows"][0]["assessment"] is None
        and report["rows"][0]["error_code"] == "TimeoutError"
    )
    assert report["summary"]["dimensions"]["grounding"]["mean"] == 5
    assert report["usage"]["estimated_cost_usd"] is None
    assert "private provider error" not in json.dumps(report)
