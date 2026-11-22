import json
from datetime import timedelta
from unittest.mock import Mock

import pytest
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.models import DEFAULT_JOB_ID
from app.schemas import InfoAnswer, InfoDecision, QualificationAnswer
from app.service import ChatService
from app.storage.models import Slot, TraceSpan


def plan_turn(service, **changes):
    service.backend.plan.return_value = service.backend.plan.return_value.model_copy(update=changes)


def test_scope_redirects_preserve_role_questions_offers_and_booking_consent(
    service, now, calendar_request
):
    from app.agents.main_agent import MIXED_SCOPE_NOTE, OFF_TOPIC_REDIRECT

    plan_turn(service, scheduling_needed=True)
    offered = service.turn("scope-calendar", "Show interview times", reference_time=now)
    stored_offers = service.state("scope-calendar")["offered_slots"]
    calls = (
        service.backend.schedule.call_count,
        service.backend.exit_label.call_count,
        service.backend.suggest_followups.call_count,
    )
    plan_turn(
        service,
        request_scope="off_topic",
        scheduling_needed=True,
        info_question="Unasked requirements",
        declined_slot_ids=[slot["id"] for slot in stored_offers],
    )
    for index, text in enumerate(
        (
            "Will it rain in Tokyo tomorrow at 10?",
            "Write an unrelated web scraper.",
            "Give me a cake recipe.",
        )
    ):
        cid = "scope-calendar" if index == 0 else f"off-topic-{index}"
        reply = service.turn(cid, text, reference_time=now)
        assert reply.response == OFF_TOPIC_REDIRECT
        assert reply.action == "continue" and reply.conversation_status == "active"
        assert (
            not reply.booking
            and not reply.offered_slots
            and not reply.source_ids
            and not reply.suggested_questions
        )
        assert not reply.advisors
    assert calls == (
        service.backend.schedule.call_count,
        service.backend.exit_label.call_count,
        service.backend.suggest_followups.call_count,
    )
    assert service.state("scope-calendar")["offered_slots"] == stored_offers

    question = "Which qualifications are required?"
    service.retriever.search = Mock(
        return_value=[
            {"id": "scope-evidence", "text": "Familiarity with SQL and NoSQL is required."}
        ]
    )
    service.backend.info.return_value = InfoAnswer(
        answer="Familiarity with SQL and NoSQL is required.",
        source_ids=["scope-evidence"],
        supported=True,
    )
    plan_turn(
        service,
        request_scope="mixed",
        recruitment_requests=[{"kind": "info", "text": question}],
        info_question="Unasked invented requirements",
        scheduling_needed=True,
    )
    reply = service.turn(
        "mixed-info", "What is the weather in Tokyo? " + question, reference_time=now
    )
    assert "SQL and NoSQL" in reply.response and MIXED_SCOPE_NOTE in reply.response
    assert reply.source_ids == ["scope-evidence"] and not reply.booking
    assert service.backend.info_decision.call_args.args[0]["messages"][-1]["content"] == question
    assert service.retriever.search.call_args.args[0] == question
    assert service.backend.schedule.call_count == calls[0]

    plan_turn(
        service,
        recruitment_requests=[{"kind": "scheduling", "text": "Please confirm this appointment."}],
        info_question=None,
    )
    booked = service.turn(
        "scope-calendar",
        "Please confirm this appointment. Also, give me a cake recipe.",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    assert booked.booking and booked.conversation_status == "active"
    assert "Your interview is confirmed" in booked.response and MIXED_SCOPE_NOTE in booked.response
    assert "info" not in booked.advisors
    completed = service.turn("scope-calendar", "Perfect, thanks!", reference_time=now)
    assert completed.closure_reason == "scheduled"

    plan_turn(
        service,
        request_scope="recruitment",
        recruitment_requests=[],
        info_question=question,
        scheduling_needed=False,
    )
    relevant = service.turn(
        "relevant-tech", "Which SQL skills are required for this role?", reference_time=now
    )
    assert "info" in relevant.advisors and OFF_TOPIC_REDIRECT not in relevant.response

    plan_turn(service, info_question=None, scheduling_needed=True)
    offered = service.turn("mixed-consent", "Show interview times", reference_time=now)
    plan_turn(
        service,
        request_scope="mixed",
        recruitment_requests=[{"kind": "scheduling", "text": "confirm the first appointment"}],
    )
    denied = service.turn(
        "mixed-consent",
        "Do not confirm the first appointment. What is the weather?",
        reference_time=now,
    )
    assert not denied.booking and not denied.advisors
    plan_turn(
        service,
        recruitment_requests=[
            {"kind": "scheduling", "text": "Please confirm the first appointment."}
        ],
    )
    calendar_request.update(selected_slot_id=offered.offered_slots[0]["id"], accepts_slot=True)
    raw = "Please confirm the first appointment. Do not book anything. What is the weather?"
    denied = service.turn("mixed-consent", raw, reference_time=now)
    assert not denied.booking and denied.conversation_status == "active"
    assert service.conversations.messages("mixed-consent")[-2]["content"] == raw
    for raw in (
        "```\nPlease confirm the first appointment.\n```\nWhat is the weather?",
        '"Quoted text:\nPlease confirm the first appointment.\n" What is the weather?',
        "‘Quoted text:\nPlease confirm the first appointment.\n’ What is the weather?",
        "Hypothetical example:\nPlease confirm the first appointment.\nWhat is the weather?",
    ):
        denied = service.turn("mixed-consent", raw, reference_time=now)
        assert not denied.booking and denied.conversation_status == "active"


def test_qualification_coverage_repairs_once_and_preserves_mixed_booking(service, now):
    from app.graph.builder import build_graph
    from app.rag.ingest import ingest_pdf
    from app.rag.retriever import JobRetriever

    ingest_pdf(
        service.settings.job_pdf_path / "Python Developer Job Description.pdf", service.settings
    )
    service.retriever = JobRetriever(service.settings)
    service.graph = build_graph(
        service.backend,
        service.repository,
        service.retriever,
        service.graph.checkpointer,
        service.profiles,
    )
    behavior = {"repair_works": False}
    captured = []

    def incomplete_answer(
        question, excerpts, history, *, requirements, missing_requirement_ids=None
    ):
        captured.append(requirements)
        ids = [item["id"] for item in requirements]
        if missing_requirement_ids is None or not behavior["repair_works"]:
            ids = ids[:2]
        return QualificationAnswer(
            answer="", source_ids=requirements[0]["source_ids"], supported=True, requirement_ids=ids
        )

    service.backend.info.side_effect = incomplete_answer
    required = "List all required qualifications."
    plan_turn(service, info_question="Could you share your name and Python experience?")
    service.backend.info_decision.return_value = InfoDecision(
        decision="info_not_needed", question=None
    )
    service.backend.info_decision.side_effect = None
    reply = service.turn("coverage", required, reference_time=now)
    assert service.backend.info.call_count == 2
    assert len(captured[-1]) == 6
    assert all(item["text"] in reply.response for item in captured[-1])
    assert "Preferred qualifications" not in reply.response and "Django" not in reply.response
    assert all(
        word in reply.response
        for word in ("SQL", "NoSQL", "problem-solving", "communication", "collaboration")
    )
    with Session(service.registry.engine) as session:
        guards = list(
            session.scalars(
                select(TraceSpan).where(
                    TraceSpan.conversation_id == "coverage", TraceSpan.component == "guardrail"
                )
            )
        )
        assert any(
            row.attributes.get("reason_code") == "source_checklist_fallback" for row in guards
        )

    behavior["repair_works"] = True
    service.backend.info.reset_mock()
    both = "Which qualifications are required, and which skills are preferred?"
    service.backend.info_decision.return_value = InfoDecision(decision="info_needed", question=both)
    plan_turn(service, info_question=both)
    reply = service.turn("coverage-both", both, reference_time=now)
    assert service.backend.info.call_count == 2 and len(captured[-1]) == 10
    assert all(item["text"] in reply.response for item in captured[-1])
    assert "Preferred qualifications" in reply.response and "Django" in reply.response

    plan_turn(service, info_question=None, scheduling_needed=True)
    offered = service.turn("coverage-booking", "Show interview times", reference_time=now)
    plan_turn(service, info_question=required)
    service.backend.info_decision.return_value = InfoDecision(
        decision="info_needed", question=required
    )
    booked = service.turn(
        "coverage-booking",
        "Please confirm this appointment. " + required,
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    assert booked.booking and booked.conversation_status == "active"
    assert "Your interview is confirmed" in booked.response
    assert {"scheduling", "info"} <= set(booked.advisors)
    assert all(item["text"] in booked.response for item in captured[-1])


def test_complete_booking_flow(service, now):
    plan_turn(service, scheduling_needed=True)
    assert service.turn("alice", reference_time=now).action == "continue"
    offered = service.turn("alice", "I would like an interview", reference_time=now)
    assert offered.action == "schedule" and len(offered.offered_slots) == 3
    booked = service.turn(
        "alice",
        "Please confirm this appointment",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    assert booked.action == "end"
    assert booked.booking and booked.conversation_status == "closed"
    assert booked.closure_reason == "scheduled"
    assert "calendar invite" not in booked.response
    service.backend.judge_quality.assert_not_called()


def test_unconfirmed_or_unoffered_selection_does_not_book(service, now, calendar_request):
    plan_turn(service, scheduling_needed=True)
    service.turn("alice", "Schedule an interview", reference_time=now)
    with pytest.raises(ValueError):
        service.turn("alice", "Book", reference_time=now, selected_slot_id=999)
    calendar_request["accepts_slot"] = True
    reply = service.turn("alice", "yes", reference_time=now)
    assert not reply.booking
    assert "Which appointment" in reply.response


def test_exit_completion_cannot_close_without_a_booking_receipt(service, now):
    service.backend.exit_label.return_value = "end"
    reply = service.turn(
        "unverified-completion", "Yes, Tuesday at 10:00 works.", reference_time=now
    )
    assert reply.action == "continue" and reply.conversation_status == "active"
    assert reply.closure_reason is None and reply.booking is None
    assert (
        reply.response
        == "Do you want to end the recruitment conversation? Please answer yes or no."
    )
    assert service.state("unverified-completion")["awaiting_opt_out_confirmation"]
    assert not reply.suggested_questions
    # A yes only authorizes closure when this particular conversation is awaiting
    # the application's own confirmation question.
    confirmed = service.turn("unverified-completion", "Yes, please.", reference_time=now)
    assert confirmed.action == "end" and confirmed.closure_reason == "opt_out"
    assert not service.state("unverified-completion")["awaiting_opt_out_confirmation"]
    for suffix, answer in (("no", "No"), ("new-topic", "What experience does this role require?")):
        cid = "uncertain-" + suffix
        service.turn(cid, "I'm unsure about my next step", reference_time=now)
        continued = service.turn(cid, answer, reference_time=now)
        assert continued.conversation_status == "active" and not continued.closure_reason
        assert "Please answer yes or no" not in continued.response
        assert not service.state(cid)["awaiting_opt_out_confirmation"]
    plan_turn(service, scheduling_needed=True)
    offered = service.turn("uncertain-slot", "Show interview times", reference_time=now)
    plan_turn(service, scheduling_needed=False)
    service.turn("uncertain-slot", "I am unsure about continuing", reference_time=now)
    assert service.state("uncertain-slot")["awaiting_opt_out_confirmation"]
    # Clicking an appointment is booking consent, even if the text is just yes.
    booked = service.turn(
        "uncertain-slot",
        "Yes",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    assert booked.booking and booked.closure_reason == "scheduled"


def test_contextual_messages_stay_active_and_opt_out_skips_scheduling_and_followups(service, now):
    contextual = (
        "Do not stop texting me",
        "What happens if I accepted another job?",
        "I am not interested in frontend work; I am interested in this Python role.",
        'My old application says "not interested", but I want to continue here.',
        "I have not accepted another job.",
        "I will not pursue this role unless the hours change.",
        "If I say I will not pursue this role, what happens?",
        '"I will not pursue this role." is an example sentence, not my decision.',
    )
    for text in contextual:
        assert service.turn("contextual", text, reference_time=now).conversation_status == "active"
    assert service.backend.exit_label.call_count == len(contextual)
    followup_calls = service.backend.suggest_followups.call_count
    reply = service.turn(
        "alice", "Please stop texting me and cancel this conversation", reference_time=now
    )
    assert reply.action == "end" and reply.conversation_status == "closed"
    assert not reply.suggested_questions
    assert "scheduling" not in reply.advisors
    assert service.backend.suggest_followups.call_count == followup_calls
    for index, text in enumerate(
        (
            "I appreciate the offer, but I will not pursue this role.",
            "Thank you for your time, but I will not pursue this position.",
            "I'd like to withdraw my application.",
            "I'll pass on this opportunity, thanks.",
        )
    ):
        declined = service.turn(f"polite-decline-{index}", text, reference_time=now)
        assert declined.action == "end" and declined.closure_reason == "opt_out", text
        assert not declined.booking and not declined.suggested_questions
        assert "scheduling" not in declined.advisors
    assert service.backend.suggest_followups.call_count == followup_calls


def test_slot_rejection_keeps_conversation_active(service, now):
    plan_turn(service, scheduling_needed=True)
    previous = service.turn("alice", "Schedule an interview", reference_time=now)
    plan_turn(service, declined_slot_ids=[slot["id"] for slot in previous.offered_slots])
    reply = service.turn("alice", "I am not available at those times", reference_time=now)
    assert reply.conversation_status == "active"
    assert not reply.booking
    assert not {item["id"] for item in reply.offered_slots} & {
        item["id"] for item in previous.offered_slots
    }


def test_booking_with_question_stays_open(service, now):
    plan_turn(service, scheduling_needed=True, info_question="What is the salary?")
    offered = service.turn(
        "alice", "Schedule an interview. What is the salary?", reference_time=now
    )
    assert offered.action == "schedule" and offered.offered_slots
    reply = service.turn(
        "alice",
        "That works. What are the role requirements?",
        reference_time=now,
        selected_slot_id=offered.offered_slots[0]["id"],
    )
    assert reply.booking
    assert reply.action == "continue"
    assert reply.conversation_status == "active"
    assert "scheduling" in reply.advisors and "info" in reply.advisors
    # A classifier can confuse completed scheduling with disinterest. A SQL
    # receipt and a pure acknowledgment must retain the scheduled closure reason.
    plan_turn(
        service,
        next_advisor="exit",
        scheduling_needed=False,
        stop_requested=True,
        closure_reason="opt_out",
    )
    service.backend.exit_label.return_value = "end"
    completed = service.turn("alice", "Perfect, thanks!", reference_time=now)
    assert completed.action == "end" and completed.closure_reason == "scheduled"
    assert completed.booking == reply.booking


def test_unknown_job_information_is_not_invented(service, now):
    plan_turn(service, info_question="What is the salary?")
    reply = service.turn("alice", "What is the salary?", reference_time=now)
    assert not reply.source_ids
    assert "recruiter" in reply.response


def add_reference_job(service):
    service.registry.source({"provider": "greenhouse", "board": "test", "company": "Example"})
    service.registry.save_job(
        {
            "job_id": "reference",
            "source_id": "greenhouse:test",
            "external_job_id": "1",
            "canonical_url": "https://example.com/1",
            "company_name": "Example",
            "title": "Python Backend Engineer",
            "description_text": "Python",
            "locations": [],
        }
    )


def test_reference_job_never_offers_assignment_calendar(service, now):
    plan_turn(service, scheduling_needed=True)
    add_reference_job(service)
    reply = service.turn(
        "reference-chat", "Schedule an interview", job_id="reference", reference_time=now
    )
    assert not reply.offered_slots and not reply.booking
    assert "reference only" in reply.response
    assert service.state("reference-chat")["job_id"] == "reference"
    with pytest.raises(ValueError, match="change jobs"):
        service.turn("reference-chat", "Hello", job_id=DEFAULT_JOB_ID, reference_time=now)


def test_availability_search_can_find_appointments_beyond_30_days(service, now):
    with Session(service.repository.engine) as session, session.begin():
        session.execute(delete(Slot))
        session.add_all(
            [
                Slot(
                    starts_at=(now + timedelta(days=40, hours=hour)).replace(tzinfo=None),
                    duration_minutes=30,
                    position="Python Dev",
                    available=True,
                )
                for hour in (1, 2, 3)
            ]
        )
    plan_turn(service, scheduling_needed=True)
    reply = service.turn("future-calendar", "Show me interview times", reference_time=now)
    assert len(reply.offered_slots) == 3 and not reply.booking
    assert all(slot["starts_at"].startswith("2026-11-29") for slot in reply.offered_slots)


def test_unavailable_preferences_offer_labelled_alternatives_and_trace_ids(
    service, now, calendar_request
):
    plan_turn(service, scheduling_needed=True)
    calendar_request.update(date_text="tomorrow", time_text="15:00")
    private = "private candidate background"
    reply = service.turn(
        "alternatives", f"{private}. Can we meet tomorrow at 15:00?", reference_time=now
    )
    assert len(reply.offered_slots) == 3 and not reply.booking
    assert "No appointments match" in reply.response and "alternatives" in reply.response
    assert service.state("alternatives")["schedule"]["status"] == "alternatives"
    with Session(service.registry.engine) as session:
        spans = list(session.scalars(select(TraceSpan).where(TraceSpan.component == "calendar")))
        searches = [item for item in spans if item.operation == "search"]
        assert {item.attributes["search_kind"] for item in searches} == {"matching", "alternatives"}
        alternative = next(
            item for item in searches if item.attributes["search_kind"] == "alternatives"
        )
        assert alternative.attributes["offered_slot_ids"] == [
            slot["id"] for slot in reply.offered_slots
        ]
        assert private not in json.dumps([item.attributes for item in spans])


def test_booking_conflict_refreshes_alternatives_then_requires_new_confirmation(service, now):
    plan_turn(service, scheduling_needed=True)
    offered = service.turn("waiting", "Show me interview times", reference_time=now)
    selected = offered.offered_slots[0]["id"]
    service.repository.book(selected, "other-candidate", "other-request", now, "UTC")
    conflict = service.turn(
        "waiting", "Confirm this appointment", reference_time=now, selected_slot_id=selected
    )
    assert not conflict.booking and conflict.conversation_status == "active"
    assert len(conflict.offered_slots) == 2
    assert selected not in {slot["id"] for slot in conflict.offered_slots}
    assert "no longer available" in conflict.response and "alternatives" in conflict.response
    assert not service.repository.get_booking("waiting", "UTC")
    confirmed = service.turn(
        "waiting",
        "Confirm this alternative",
        reference_time=now,
        selected_slot_id=conflict.offered_slots[0]["id"],
    )
    assert confirmed.booking and confirmed.conversation_status == "closed"
    assert confirmed.booking["slot"]["id"] != selected


def test_checkpoint_survives_service_restart(now, backend):
    class EmptyRetriever:
        def search(self, _query, job_id=None):
            return []

    settings = Settings(_env_file=None)
    first = ChatService(settings, backend=backend, retriever=EmptyRetriever())
    first.turn("persisted", reference_time=now)
    first.turn("persisted", "I have used Python for two years", reference_time=now)
    original = first.state("persisted")["messages"]
    first.close()
    second = ChatService(settings, backend=backend, retriever=EmptyRetriever())
    try:
        # Simulate the hosted AdminShutdown failure on a pooled idle checkpoint
        # connection. The checkout health check replaces it before any state read.
        import psycopg

        with second._checkpoint_pool.connection() as connection:
            dead_pid = connection.info.backend_pid
        with psycopg.connect(settings.postgres_dsn, autocommit=True) as admin:
            assert admin.execute("SELECT pg_terminate_backend(%s)", (dead_pid,)).fetchone()[0]
        assert second.state("persisted")["messages"] == original
        with second._checkpoint_pool.connection() as connection:
            assert connection.info.backend_pid != dead_pid
        reply = second.turn("persisted", "Please stop texting me", reference_time=now)
        assert reply.action == "end"
        assert len(second.state("persisted")["messages"]) == len(original) + 2
    finally:
        second.close()


def routed_plan(service, choice, **changes):
    return service.backend.plan.return_value.model_copy(update={"next_advisor": choice, **changes})


def test_main_reviews_info_advice_before_requesting_scheduling(service, now):
    first = "info"
    second = "scheduling"
    plans = [
        routed_plan(
            service,
            name,
            info_question="Required skills?" if name == "info" else None,
            scheduling_needed=True,
        )
        for name in [first, first, "respond"]
    ]
    service.backend.plan.side_effect = plans
    service.retriever.search = Mock(return_value=[{"id": "chunk-1", "text": "Python and SQL"}])
    service.backend.info.return_value = InfoAnswer(
        answer="Python and SQL are required.", source_ids=["chunk-1"], supported=True
    )
    response = service.turn(
        "review", "What skills are required, and can I interview?", reference_time=now
    )
    assert response.advisors == [first, second]
    assert "Python and SQL" in response.response and response.offered_slots
    assert not response.booking
    service.backend.exit_label.assert_not_called()
    reviewed = service.backend.plan.call_args_list[1].args[0]
    assert reviewed["advisor_results"][0]["advisor"] == first
    assert reviewed["advisor_results"][0]["decision"] in {"info_needed", "schedule"}
    state = service.state("review")
    assert [item["next_advisor"] for item in state["routing_history"]] == [first, second, "respond"]
    assert state["routing_history"][1]["rerouted_from"] == "info"
    assert len(state["messages"]) == 2
    assert [message["role"] for message in state["messages"]] == ["user", "assistant"]


def test_main_can_reconsult_info_for_a_distinct_unanswered_question(service, now):
    service.backend.plan.side_effect = [
        routed_plan(service, "info", info_question="Responsibilities?"),
        routed_plan(service, "info", info_question="Required skills?"),
        routed_plan(service, "respond"),
    ]
    service.retriever.search = Mock(
        side_effect=[
            [{"id": "responsibilities", "text": "Build Python services."}],
            [{"id": "skills", "text": "SQL is required."}],
        ]
    )
    service.backend.info.side_effect = [
        InfoAnswer(
            answer="Build Python services.", source_ids=["responsibilities"], supported=True
        ),
        InfoAnswer(answer="SQL is required.", source_ids=["skills"], supported=True),
    ]
    response = service.turn(
        "two-questions", "What are the responsibilities and required skills?", reference_time=now
    )
    assert response.advisors == ["info", "info"]
    assert set(response.source_ids) == {"responsibilities", "skills"}
    assert "Build Python services" in response.response and "SQL is required" in response.response
    assert service.state("two-questions")["routing_stop_reason"] is None
    assert len(service.state("two-questions")["messages"]) == 2


def test_info_not_needed_advice_skips_retrieval_and_scheduling(service, now):
    advisor = "info"
    service.backend.plan.side_effect = [
        routed_plan(service, advisor),
        routed_plan(service, "respond"),
    ]
    service.retriever.search = Mock()
    service.repository.available = Mock()
    service.repository.book = Mock()
    service.backend.info_decision.side_effect = None
    service.backend.info_decision.return_value = InfoDecision(
        decision="info_not_needed", question=None
    )
    response = service.turn("not-needed", "I used Python on a project.", reference_time=now)
    assert response.advisors == [advisor] and not response.booking and not response.offered_slots
    service.retriever.search.assert_not_called()
    service.repository.available.assert_not_called()
    service.repository.book.assert_not_called()
    service.backend.info.assert_not_called()
    service.backend.schedule.assert_not_called()
    assert service.state("not-needed")["advisor_results"][0]["decision"] == "info_not_needed"
    # A new, explicit role question cannot be erased by a helper's negative decision.
    service.backend.plan.side_effect = [
        routed_plan(
            service, "info", info_question="What are your responsibilities for this position?"
        ),
        routed_plan(service, "respond"),
    ]
    service.retriever.search.return_value = []
    service.turn("explicit-question", "What would my responsibilities be?", reference_time=now)
    service.retriever.search.assert_called_once()
    assert service.retriever.search.call_args.args[0] == "What would my responsibilities be?"
    assert service.state("explicit-question")["advisor_results"][0]["decision"] == "info_needed"


def test_consultation_loop_stops_without_repeating_unchanged_work(service, now):
    reason = "consultation_limit"
    plan = routed_plan(service, "info", info_question="Required skills?")
    service.backend.plan.side_effect = lambda state: plan
    service.retriever.search = Mock(return_value=[])
    service.settings.max_advisor_consultations = 1
    response = service.turn("bounded", "What skills are required?", reference_time=now)
    state = service.state("bounded")
    assert response.advisors == ["info"] and not response.error
    assert state["routing_stop_reason"] == reason
    assert state["consultation_count"] == 1
    assert service.retriever.search.call_count == 1
    assert len(state["messages"]) == 2
