import json

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.graph.builder import build_graph
from app.storage.models import ConversationMessage, TraceSpan
from app.suggestions import STARTER_QUESTIONS


def test_followups_save_export_and_click_as_a_normal_message(service, now, capfd):
    first = service.turn("suggestions", "Hello", request_id="answer-1", reference_time=now)
    assert first.suggestion_source == "model" and len(first.suggested_questions) == 3
    context = service.backend.suggest_followups.call_args.args[0]
    assert context["latest_answer"] == first.response
    assert context["history"][-1] == {"role": "assistant", "content": first.response}
    assert context["job"]["job_id"] == service.state("suggestions")["job_id"]
    assert "candidate_profile" not in context and "extracted_text" not in context
    chosen = first.suggested_questions[0]
    selection = {"suggestion_id": chosen.id, "answer_request_id": "answer-1"}
    second = service.turn(
        "suggestions", chosen.text, request_id="click-1", suggestion=selection, reference_time=now
    )
    assert len(second.suggested_questions) == 3
    assert chosen.text not in [item.text for item in second.suggested_questions]
    assert service.conversations.messages("suggestions")[2]["content"] == chosen.text
    calls = service.backend.suggest_followups.call_count
    assert (
        service.turn(
            "suggestions",
            chosen.text,
            request_id="click-1",
            suggestion=selection,
            reference_time=now,
        )
        == second
    )
    assert service.backend.suggest_followups.call_count == calls
    turns = json.loads(service.settings.conversation_export_path.read_text())[0]["turns"]
    assert len(turns) == 4
    assert turns[1]["suggested_questions"] == [q.model_dump() for q in first.suggested_questions]
    assert turns[2]["suggestion"] == selection
    with Session(service.registry.engine) as session:
        spans = list(session.scalars(select(TraceSpan)))
        generation = [row for row in spans if row.operation == "suggest_followups"]
        assert len(generation) == 2
        selected = [row for row in spans if row.operation == "select_suggestion"]
        assert len(selected) == 1 and selected[0].attributes == selection
        parents = {row.span_id: row.trace_id for row in spans}
        assert all(parents[row.parent_span_id] == row.trace_id for row in generation + selected)
        assert chosen.text not in json.dumps([row.attributes for row in spans])
    assert chosen.text not in capfd.readouterr().err


@pytest.mark.parametrize("invalid", ["provider", "duplicate", "perspective"])
def test_failed_or_invalid_generation_keeps_the_answer_and_three_fallbacks(service, now, invalid):
    if invalid == "provider":
        service.backend.suggest_followups.side_effect = TimeoutError("private provider details")
    elif invalid == "perspective":
        service.backend.suggest_followups.return_value = {
            "questions": [
                "What dates or times work for you to schedule an interview?",
                "Would you like to discuss interview preparation with the recruiter?",
                "Are there topics you want me to clarify with the recruiter?",
            ]
        }
    else:
        questions = [
            "Which skills would I use daily?",
            "How could I show my relevant experience?",
            "What can you tell me about the team?",
        ]
        questions[1] = questions[0]
        service.backend.suggest_followups.return_value = {"questions": questions}
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(
        update={"info_question": "What skills are required for this role?"}
    )
    result = service.turn(
        "fallback", "What skills are required for this role?", request_id="one", reference_time=now
    )
    assert not result.error and result.response
    assert result.suggestion_source == "fallback"
    texts = [item.text for item in result.suggested_questions]
    assert len(texts) == len(set(texts)) == 3
    assert "Which skills would I use most often in this role?" in texts
    assert "What skills are required for this role?" not in texts


def test_resume_after_composition_finishes_followups_without_repeating_advisors(
    service, now, monkeypatch
):
    from app.graph import followups

    original = followups.identified_questions
    attempts = {"count": 0}

    def interrupt_once(*args):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("worker interrupted before suggestion checkpoint")
        return original(*args)

    monkeypatch.setattr(followups, "identified_questions", interrupt_once)
    with pytest.raises(OSError):
        service.turn("resume-suggestions", "Hello", request_id="same", reference_time=now)
    assert service.graph.get_state(service._config("resume-suggestions")).next == (
        "suggest_followups",
    )
    plan_calls = service.backend.plan.call_count
    service.graph = build_graph(
        service.backend,
        service.repository,
        service.retriever,
        service.graph.checkpointer,
        service.profiles,
    )
    result = service.turn("resume-suggestions", "Hello", request_id="same", reference_time=now)
    assert len(result.suggested_questions) == 3
    assert service.backend.plan.call_count == plan_calls
    assert len(service.state("resume-suggestions")["messages"]) == 2
    assert len(service.conversations.messages("resume-suggestions")) == 2


def test_starters_and_followups_reject_tampered_stale_or_other_conversation_clicks(service, now):
    starter = STARTER_QUESTIONS[0]
    selection = {"suggestion_id": starter.id, "answer_request_id": None}
    with pytest.raises(ValueError, match="does not belong"):
        service.turn("tampered", "Different text", suggestion=selection, reference_time=now)
    first = service.turn("clicks", starter.text, suggestion=selection, reference_time=now)
    chosen = first.suggested_questions[0]
    answer_id = service.state("clicks")["request_id"]
    selection = {"suggestion_id": chosen.id, "answer_request_id": answer_id}
    with pytest.raises(ValueError, match="no longer available"):
        service.turn("other-conversation", chosen.text, suggestion=selection, reference_time=now)
    service.turn("clicks", "Tell me more", reference_time=now)
    with pytest.raises(ValueError, match="no longer available"):
        service.turn("clicks", chosen.text, suggestion=selection, reference_time=now)
    with Session(service.registry.engine) as session:
        assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 4


def test_closed_conversation_and_evaluation_skip_generation(service, now):
    closed = service.turn("ended", "Stop", reference_time=now)
    assert closed.conversation_status == "closed" and not closed.suggested_questions
    service.backend.suggest_followups.assert_not_called()
    result = service.evaluate_prefix([{"role": "user", "content": "Hello"}], now)
    assert result.response and not result.suggested_questions
    service.backend.suggest_followups.assert_not_called()
