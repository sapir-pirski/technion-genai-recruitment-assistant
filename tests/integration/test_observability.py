import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import httpx
import pytest
from langchain_openai import ChatOpenAI
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.backends.common import prompt
from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from app.observability.logging import JsonFormatter
from app.observability.tracing import EmbeddingClient
from app.service import ChatService
from app.storage.models import Booking, ConversationMessage, TraceSpan


def test_history_export_failed_turn_and_replay(benchmark_dataset, service, now, capfd):
    original_benchmark = benchmark_dataset.load_conversations()
    private_text = "private candidate history"
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(
        update={"info_question": "requirements"}
    )
    service.retriever.search = Mock(
        return_value=[{"id": "chunk-1", "text": "private source excerpt"}]
    )
    first = service.turn("audit", private_text, request_id="first", reference_time=now)
    first_calls = service.backend.plan.call_count
    assert service.turn("audit", private_text, request_id="first", reference_time=now) == first
    assert service.backend.plan.call_count == first_calls
    service.backend.plan.side_effect = TimeoutError("secret exception contents")
    failed = service.turn("audit", "Try again", request_id="failed", reference_time=now)
    assert failed.error == "TimeoutError"
    assert service.turn("audit", "Try again", request_id="failed", reference_time=now) == failed
    assert service.backend.plan.call_count == first_calls + 1
    with Session(service.registry.engine) as session:
        messages = list(
            session.scalars(select(ConversationMessage).order_by(ConversationMessage.sequence_no))
        )
        spans = list(session.scalars(select(TraceSpan)))
        assert len(messages) == 4
        assert messages[0].occurred_at == now
        roots = [item for item in spans if item.parent_span_id is None]
        assert len(roots) == 2
        assert {item.status for item in roots} == {"complete", "failed"}
        identities = {item.span_id: item.trace_id for item in spans}
        assert all(
            item.parent_span_id is None or identities[item.parent_span_id] == item.trace_id
            for item in spans
        )
        retrieval = next(item for item in spans if item.component == "retrieval")
        assert retrieval.attributes["source_ids"] == ["chunk-1"]
        assert all(item.input_tokens is None for item in spans)
        assert private_text not in json.dumps([item.attributes for item in spans])
    export = json.loads(service.settings.conversation_export_path.read_text())
    assert len(export) == 1 and len(export[0]["turns"]) == 4
    assert export[0]["candidate_phone"] is None and export[0]["recruiter_phone"] is None
    assert [item["speaker"] for item in export[0]["turns"]] == [
        "candidate",
        "recruiter",
        "candidate",
        "recruiter",
    ]
    assert [item["turn_id"] for item in export[0]["turns"]] == [1, 2, 3, 4]
    assert all(item["label"] is None for item in export[0]["turns"])
    assert export[0]["turns"][1]["predicted_action"] == first.action
    assert service.settings.conversation_export_path.stat().st_mode & 0o777 == 0o600
    assert benchmark_dataset.load_conversations() == original_benchmark
    diagnostics = capfd.readouterr().err
    assert private_text not in diagnostics and "secret exception contents" not in diagnostics
    assert all(isinstance(json.loads(line), dict) for line in diagnostics.splitlines())
    record = logging.LogRecord(
        "app", logging.ERROR, "", 1, "secret prompt %s", ("secret key",), None
    )
    record.api_key = "secret key"
    assert "secret" not in JsonFormatter().format(record)


def test_concurrent_workers_deduplicate_and_export_every_conversation(backend, now):
    settings = Settings(_env_file=None)
    workers = [
        ChatService(settings, backend=backend, retriever=Mock(search=Mock(return_value=[])))
        for _ in range(2)
    ]
    try:

        def turn(index, conversation="same"):
            return workers[index].turn(
                conversation, "Hello", request_id="one-request", reference_time=now
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(turn, [0, 1]))
        assert results[0] == results[1] and backend.plan.call_count == 2
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(turn, index, f"worker-{index}") for index in range(2)]
            for future in futures:
                future.result()
        export = json.loads(settings.conversation_export_path.read_text())
        assert len(export) == 3 and all(len(row["turns"]) == 2 for row in export)
        with Session(workers[0].registry.engine) as session:
            assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 6
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(TraceSpan)
                    .where(TraceSpan.parent_span_id.is_(None))
                )
                == 3
            )
    finally:
        for worker in workers:
            worker.close()


def test_diagnostics_failures_preserve_booking_and_durable_receipt(service, now, capfd):
    target = "app.observability.tracing.Session"
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    offered = service.turn("booking", "Schedule", request_id="offer", reference_time=now)
    arguments = dict(
        request_id="confirm", reference_time=now, selected_slot_id=offered.offered_slots[0]["id"]
    )
    with patch(target, side_effect=OSError("secret error details")):
        booked = service.turn("booking", "Confirm", **arguments)
    assert booked.booking and booked.conversation_status == "closed"
    calls = service.backend.plan.call_count
    assert service.turn("booking", "Confirm", **arguments) == booked
    assert service.backend.plan.call_count == calls
    with Session(service.registry.engine) as session:
        assert session.scalar(select(func.count()).select_from(Booking)) == 1
        assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 4
    export = json.loads(service.settings.conversation_export_path.read_text())
    assert export[0]["status"] == "closed" and len(export[0]["turns"]) == 4
    assert not list(service.settings.conversation_export_path.parent.glob(".conversations-*.tmp"))
    assert "secret error details" not in capfd.readouterr().err


def test_checkpoint_receipt_recovers_after_message_write_failure(service, now):
    with patch.object(service.conversations, "complete", side_effect=RuntimeError("write failed")):
        with pytest.raises(RuntimeError, match="write failed"):
            service.turn("recover", "Hello", request_id="same-request", reference_time=now)
    calls = service.backend.plan.call_count
    assert calls == 2
    response = service.turn("recover", "Hello", request_id="same-request", reference_time=now)
    assert response.response and service.backend.plan.call_count == calls
    export = json.loads(service.settings.conversation_export_path.read_text())
    assert len(export[0]["turns"]) == 2


@pytest.mark.parametrize("case", ["parse_failure", "missing"])
def test_api_usage_is_preserved_before_parsing_and_missing_counts_stay_null(service, now, case):
    settings = service.settings.model_copy(
        update={"openai_api_key": service.settings.openai_api_key}
    )
    backend = OpenAIBackend(settings)
    content = (
        json.dumps(service.backend.plan.return_value.model_dump())
        if case != "parse_failure"
        else "not valid json"
    )
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": settings.chat_model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    if case != "missing":
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 12,
            "total_tokens": 112,
            "prompt_tokens_details": {"cached_tokens": 20},
        }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        backend.chat = ChatOpenAI(
            model=settings.chat_model, api_key="test-only-key", http_client=client
        )
        try:
            with service.traces.trace(None, "usage-test", component="evaluation"):
                with Session(service.registry.engine) as session:
                    active = session.scalar(
                        select(TraceSpan).where(TraceSpan.request_id == "usage-test")
                    )
                    assert active.status == "running" and active.finished_at is None
                backend.plan({"messages": [], "reference_time": now.isoformat(), "timezone": "UTC"})
        except Exception:
            if case != "parse_failure":
                raise
        else:
            assert case != "parse_failure"
    with service.traces.trace(None, "embedding-test", component="ingestion"):
        EmbeddingClient(
            Mock(create=Mock(return_value={"usage": {"prompt_tokens": 8, "total_tokens": 8}})),
            "text-embedding-3-small",
        ).create(input=["test"])
    with Session(service.registry.engine) as session:
        model = session.scalar(select(TraceSpan).where(TraceSpan.component == "llm"))
        assert model.input_tokens == (None if case == "missing" else 100)
        assert model.output_tokens == (None if case == "missing" else 12)
        assert model.cached_input_tokens == (None if case == "missing" else 20)
        assert model.status == ("failed" if case == "parse_failure" else "complete")
        assert model.attributes["prompt_name"] == "main"
        assert (
            model.attributes["prompt_sha256"] == hashlib.sha256(prompt("main").encode()).hexdigest()
        )
        assert model.attributes["instruction_role"] == "system"
        assert prompt("main") not in json.dumps(model.attributes)
        root = session.scalar(
            select(TraceSpan).where(
                TraceSpan.request_id == "usage-test", TraceSpan.parent_span_id.is_(None)
            )
        )
        assert root.input_tokens is None
        embedding = session.scalar(select(TraceSpan).where(TraceSpan.component == "embedding"))
        assert embedding.input_tokens == 8 and embedding.output_tokens is None
