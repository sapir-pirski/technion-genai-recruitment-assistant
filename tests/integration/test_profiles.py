import json

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.graph.builder import build_graph
from app.schemas import ChatProfileUpdate, ProfileSections
from app.storage.models import CandidateProfile, CandidateSource, ConversationMessage, TraceSpan


def sections(*, experience=(), skills=(), name=None):
    return ProfileSections(
        name=name, experience=list(experience), skills=list(skills), education=[], projects=[]
    )


def test_all_sources_are_automatic_idempotent_and_not_fake_messages(service, now):
    values = {
        "introduction": "3.5 years of Python experience.",
        "linkedin_url": "https://linkedin.com/in/alex-example?trk=test",
        "linkedin_text": "Skills: Python, SQL",
        "file_name": "cv.txt",
        "file_data": b"Projects with Django",
    }
    service.backend.extract_profile.side_effect = [
        sections(experience=["3.5 years of Python experience."]),
        sections(skills=["Python", "SQL"]),
        sections(experience=["Projects with Django"], skills=["Django"]),
    ]
    first = service.turn(
        "auto", "Hello", context_inputs=values, request_id="first", reference_time=now
    )
    state = service.state("auto")
    assert state["profile_status"] == "ready"
    assert state["candidate_profile"]["sections"]["skills"] == ["Python", "SQL", "Django"]
    assert service.backend.plan.call_args_list[0].args[0]["candidate_profile"]["sections"][
        "skills"
    ] == ["Python", "SQL", "Django"]
    assert (
        service.turn("auto", "Hello", context_inputs=values, request_id="first", reference_time=now)
        == first
    )
    service.turn("auto", "What next?", context_inputs=values, reference_time=now)
    assert service.backend.extract_profile.call_count == 3
    service.backend.schedule.assert_not_called()
    export = json.loads(service.settings.conversation_export_path.read_text())[0]
    assert export["turns"][0]["text"] == "Hello" and len(export["turns"]) == 4
    assert len(export["sources"]) == 4 and export["profile"]["status"] == "ready"
    assert all("input_data" not in source for source in export["sources"])
    with Session(service.registry.engine) as session:
        assert session.scalar(select(func.count()).select_from(CandidateSource)) == 4
        assert session.scalar(select(func.count()).select_from(CandidateProfile)) == 1
        assert all(row.input_data is None for row in session.scalars(select(CandidateSource)))
        spans = list(
            session.scalars(select(TraceSpan).where(TraceSpan.component == "candidate_profile"))
        )
        assert spans and all(row.conversation_id == "auto" for row in spans)
        assert all(
            "linkedin.com" not in json.dumps(row.attributes)
            and "3.5 years" not in json.dumps(row.attributes)
            for row in spans
        )


def test_changes_removals_and_same_content_in_other_conversation_are_isolated(service, now):
    service.backend.extract_profile.side_effect = lambda text: sections(
        skills=["Python"] if "Python" in text else ["SQL"]
    )
    service.turn("one", "Hi", context_inputs={"introduction": "Python"}, reference_time=now)
    old = service.state("one")["profile_version"]
    service.turn("one", "Changed", context_inputs={"introduction": "SQL"}, reference_time=now)
    assert service.state("one")["profile_version"] != old
    assert service.state("one")["candidate_profile"]["sections"]["skills"] == ["SQL"]
    service.turn("two", "Hi", context_inputs={"introduction": "SQL"}, reference_time=now)
    assert service.backend.extract_profile.call_count == 3
    foreign = service.state("two")["profile_source_ids"]
    with pytest.raises(ValueError, match="does not belong"):
        service.profiles.sources("one", foreign)
    service.turn("one", "Cleared", context_inputs={}, reference_time=now)
    assert service.state("one")["profile_status"] == "empty"
    assert service.state("one")["candidate_profile"]["sections"]["skills"] == []
    with Session(service.registry.engine) as session:
        assert (
            session.scalar(
                select(func.count())
                .select_from(CandidateProfile)
                .where(CandidateProfile.is_current.is_(True))
            )
            == 2
        )


def test_partial_failures_and_url_only_do_not_block_chat(service, now):
    service.backend.extract_profile.return_value = sections(skills=["Python"])
    result = service.turn(
        "partial",
        "What next?",
        context_inputs={
            "introduction": "Python",
            "file_name": "broken.pdf",
            "file_data": b"broken",
        },
        reference_time=now,
    )
    assert not result.error and service.state("partial")["profile_status"] == "partial"
    assert service.state("partial")["profile_issues"][0]["error_code"] == "InvalidPDF"
    calls = service.backend.extract_profile.call_count
    result = service.turn(
        "url",
        "Hi",
        context_inputs={"linkedin_url": "https://www.linkedin.com/in/alex-example/"},
        reference_time=now,
    )
    assert not result.error and service.backend.extract_profile.call_count == calls
    assert service.state("url")["candidate_profile"]["references"]


def test_resume_uses_frozen_snapshot_and_completed_facts(service, now, monkeypatch):
    service.backend.extract_profile.return_value = sections(skills=["Python"])
    save = service.profiles.save_profile
    attempts = {"count": 0}

    def fail_once(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("simulated interruption")
        return save(*args, **kwargs)

    monkeypatch.setattr(service.profiles, "save_profile", fail_once)
    with pytest.raises(OSError):
        service.turn(
            "resume",
            "Hi",
            context_inputs={"introduction": "Python"},
            request_id="r1",
            reference_time=now,
        )
    # A new graph instance simulates a worker restart using the same checkpointer and SQL state.
    service.graph = build_graph(
        service.backend,
        service.repository,
        service.retriever,
        service.graph.checkpointer,
        service.profiles,
    )
    result = service.turn(
        "resume",
        "Hi",
        context_inputs={"introduction": "CHANGED UNSENT INPUT"},
        request_id="r1",
        reference_time=now,
    )
    assert not result.error and service.backend.extract_profile.call_count == 1
    assert service.state("resume")["candidate_profile"]["sections"]["skills"] == ["Python"]
    with Session(service.registry.engine) as session:
        assert session.scalar(select(func.count()).select_from(ConversationMessage)) == 2


def test_opt_out_does_not_read_supplied_sources(service, now):
    result = service.turn(
        "stop",
        "Please stop texting me",
        context_inputs={
            "introduction": "Python",
            "file_name": "cv.txt",
            "file_data": b"private text",
        },
        reference_time=now,
    )
    assert result.action == "end"
    service.backend.extract_profile.assert_not_called()


def test_transient_extraction_failure_is_retried_on_next_message(service, now):
    service.backend.extract_profile.side_effect = [TimeoutError(), sections(skills=["Python"])]
    inputs = {"introduction": "Python"}
    first = service.turn("retry-source", "Hi", context_inputs=inputs, reference_time=now)
    assert not first.error and service.state("retry-source")["profile_status"] == "failed"
    assert service.state("retry-source")["profile_issues"][0]["retryable"]
    second = service.turn("retry-source", "Try again", context_inputs=inputs, reference_time=now)
    assert not second.error and service.state("retry-source")["profile_status"] == "ready"
    assert service.backend.extract_profile.call_count == 2
    assert not service.state("retry-source")["profile_issues"]


def test_chat_facts_corrections_confirmations_and_resume_are_durable(service, now, monkeypatch):
    service.backend.extract_profile.return_value = sections(
        experience=["4 years of Python experience"], skills=["Python"]
    )
    service.turn(
        "chat-facts",
        "Hello",
        context_inputs={"introduction": "4 years of Python experience. Python"},
        reference_time=now,
    )
    original = service.state("chat-facts")["candidate_profile"]
    old_id = next(item["fact_id"] for item in original["evidence"] if item["field"] == "experience")
    service.backend.extract_chat_profile.return_value = ChatProfileUpdate(
        facts=sections(experience=["2 years of Python experience"]),
        # A changed claim alone cannot authorize a model-proposed replacement.
        superseded_fact_ids=[old_id],
        confirmed_fact_ids=["unknown-fact"],
    )
    response = service.turn(
        "chat-facts", "I have 2 years of Python experience.", request_id="claim", reference_time=now
    )
    state = service.state("chat-facts")
    assert state["candidate_profile"]["conflicts"][0]["field"] == "python_experience"
    assert not state["candidate_profile"]["verification"]["confirmed_fact_ids"]
    assert "different Python experience claims" in response.response
    chat_fact = next(
        item for item in state["candidate_profile"]["evidence"] if item["origin"] == "chat"
    )
    assert chat_fact["message_id"] and chat_fact["request_id"] == "claim"
    calls = service.backend.extract_chat_profile.call_count
    assert (
        service.turn(
            "chat-facts",
            "I have 2 years of Python experience.",
            request_id="claim",
            reference_time=now,
        )
        == response
    )
    assert service.backend.extract_chat_profile.call_count == calls
    service.backend.extract_chat_profile.return_value = ChatProfileUpdate(
        # An added sentence stop can be removed without changing the source words.
        facts=sections(experience=["2 years of Python experience."]),
        superseded_fact_ids=[old_id],
        confirmed_fact_ids=[],
    )
    save = service.profiles.save_profile
    with monkeypatch.context() as patch:
        patch.setattr(
            service.profiles,
            "save_profile",
            lambda *args: (_ for _ in ()).throw(OSError("injected profile publication failure")),
        )
        with pytest.raises(OSError):
            service.turn(
                "chat-facts",
                "Actually, I have 2 years of Python experience, not 4.",
                request_id="correct",
                reference_time=now,
            )
    calls = service.backend.extract_chat_profile.call_count
    service.turn(
        "chat-facts",
        "Actually, I have 2 years of Python experience, not 4.",
        request_id="correct",
        reference_time=now,
    )
    assert service.backend.extract_chat_profile.call_count == calls
    profile = service.state("chat-facts")["candidate_profile"]
    assert profile["sections"]["experience"] == ["2 years of Python experience"]
    assert not profile["conflicts"]
    assert (
        next(item for item in profile["evidence"] if item["fact_id"] == old_id)["status"]
        == "superseded"
    )
    service.backend.extract_chat_profile.return_value = ChatProfileUpdate(
        # An unsupported new fact must not discard an independently valid confirmation.
        facts=sections(skills=["Kubernetes"]),
        superseded_fact_ids=[],
        confirmed_fact_ids=[chat_fact["fact_id"]],
    )
    service.turn("chat-facts", "Yes, I confirm 2 years of Python experience.", reference_time=now)
    profile = service.state("chat-facts")["candidate_profile"]
    assert chat_fact["fact_id"] in profile["verification"]["confirmed_fact_ids"]
    assert "Kubernetes" not in profile["sections"]["skills"]
    stored = service.profiles.profile(service.state("chat-facts")["profile_version"], "chat-facts")
    assert stored["data"] == profile
    assert service.profiles.save_profile == save
