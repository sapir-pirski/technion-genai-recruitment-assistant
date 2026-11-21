import json

from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.backends.common import model_context
from app.config import DEFAULT_JOB_ID
from app.scheduling.models import Booking
from app.storage.models import ConversationMessage
from app.workflow.state import ConversationState


def test_background_stays_in_history_without_profile_or_followup_processing(service, now):
    text = "I have four years of Python experience and build Flask APIs."
    events = []
    reply = service.turn(
        "background", text, request_id="background-1", reference_time=now, progress=events.append
    )
    assert not reply.error and reply.action == "continue"
    state = service.state("background")
    context = model_context(state)
    assert context["history"][0] == {"role": "user", "content": text}
    assert "candidate_profile" not in context and "candidate_profile" not in state
    assert "suggested_questions" not in reply.model_dump()
    assert service.backend.plan.call_count == 2 and service.backend.exit_label.call_count == 1
    assert sum(event["stage"] == "answer_ready" for event in events) == 1
    assert not any("profile" in event["stage"] or "followup" in event["stage"] for event in events)
    components = {item.component for item in service.traces.last_trace}
    assert "candidate_profile" not in components and "app.graph.followups" not in components
    calls = list(service.backend.method_calls)
    assert service.turn("background", text, request_id="background-1", reference_time=now) == reply
    assert service.backend.method_calls == calls
    exported = json.loads(service.settings.conversation_export_path.read_text())
    assert "suggested_questions" not in exported[0]["turns"][-1]


def test_legacy_booking_receipts_remain_readable_without_reintroducing_question_buttons(
    service, now
):
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    offered = service.turn(
        "legacy-booking", "Show interview times", request_id="offer", reference_time=now
    )
    booked = service.turn(
        "legacy-booking",
        "Confirm",
        request_id="confirm",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    legacy = {
        **booked.model_dump(),
        "suggested_questions": [{"id": "old-question", "text": "What should I prepare?"}],
        "suggestion_source": "model",
    }
    with Session(service.registry.engine) as session, session.begin():
        row = session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.conversation_id == "legacy-booking",
                ConversationMessage.request_id == "confirm",
                ConversationMessage.role == "assistant",
            )
        )
        row.response_data = legacy
    calls = list(service.backend.method_calls)
    replayed = service.turn("legacy-booking", "Confirm", request_id="confirm", reference_time=now)
    assert replayed.booking == booked.booking and replayed.conversation_status == "closed"
    assert "suggested_questions" not in replayed.model_dump()
    assert service.backend.method_calls == calls
    with Session(service.registry.engine) as session:
        assert session.scalar(select(func.count()).select_from(Booking)) == 1
        assert (
            session.scalar(
                select(ConversationMessage.response_data).where(
                    ConversationMessage.request_id == "confirm",
                    ConversationMessage.role == "assistant",
                )
            )
            == legacy
        )


def test_pending_retired_profile_node_preserves_the_candidates_chat_message(service, now):
    class LegacyState(ConversationState, total=False):
        candidate_profile: dict

    graph = StateGraph(LegacyState)
    graph.add_node("profile_ingestion", lambda state: {})
    graph.add_edge(START, "profile_ingestion")
    graph.add_edge("profile_ingestion", END)
    legacy = graph.compile(
        checkpointer=service.graph.checkpointer, interrupt_before=["profile_ingestion"]
    )
    text = "I build Python APIs."
    legacy.invoke(
        {
            "conversation_id": "retired-node",
            "request_id": "same-request",
            "job_id": DEFAULT_JOB_ID,
            "messages": [{"role": "user", "content": text}],
            "reference_time": now.isoformat(),
            "timezone": "UTC",
            "result": {},
            "conversation_status": "active",
            "candidate_profile": {"sections": {"skills": ["OBSOLETE PROFILE CLAIM"]}},
        },
        service._config("retired-node"),
    )
    reply = service.turn("retired-node", text, request_id="same-request", reference_time=now)
    assert not reply.error and reply.response
    state = service.state("retired-node")
    assert [message["content"] for message in state["messages"] if message["role"] == "user"] == [
        text
    ]
    assert "candidate_profile" not in model_context(state)
    assert "OBSOLETE PROFILE CLAIM" not in str(service.backend.plan.call_args_list)
