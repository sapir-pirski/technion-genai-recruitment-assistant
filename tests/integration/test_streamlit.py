from streamlit.testing.v1 import AppTest

from app.config import ROOT
from app.suggestions import STARTER_QUESTIONS

# UI checks use this test module's disposable PostgreSQL databases.
UI_TIMEOUT = 60


def test_startup_recovers_automatically_and_reports_configuration_safely(monkeypatch):
    from psycopg.errors import AdminShutdown, InvalidPassword

    from app.service import ChatService

    original_state, original_init = ChatService.state, ChatService.__init__
    attempts, instances = [], []
    failure = {"remaining": 2, "permanent": False}

    def construct(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)

    def state(self, conversation_id):
        attempts.append(conversation_id)
        if failure["permanent"]:
            raise InvalidPassword("Fictitious private DSN/password must not be shown")
        if failure["remaining"]:
            failure["remaining"] -= 1
            raise AdminShutdown("Synthetic startup disconnection")
        return original_state(self, conversation_id)

    monkeypatch.setattr(ChatService, "__init__", construct)
    monkeypatch.setattr(ChatService, "state", state)
    test = AppTest.from_file(str(ROOT / "streamlit_app/streamlit_main.py")).run(timeout=UI_TIMEOUT)
    assert not test.exception and not test.error and test.chat_input
    assert len(attempts) == 3 and len(set(attempts)) == 1 and len(instances) == 1
    conversation = test.session_state.conversation_id
    test.text_area[0].set_value("Startup Example. Python and SQL experience.").run(
        timeout=UI_TIMEOUT
    )
    test.text_input[0].set_value("https://www.linkedin.com/in/startup-example/").run(
        timeout=UI_TIMEOUT
    )
    test.file_uploader[0].upload(
        "startup.txt", b"Startup Example. Python experience.", "text/plain"
    ).run(timeout=UI_TIMEOUT)
    test.run(timeout=UI_TIMEOUT)
    assert len(instances) == 1 and test.session_state.conversation_id == conversation
    failure["permanent"] = True
    calls = len(attempts)
    test.run(timeout=UI_TIMEOUT)
    assert not test.exception and len(attempts) == calls + 1
    assert test.button(key="retry_startup")
    assert "contact the app owner" in test.error[0].value
    assert "private DSN" not in str(test)
    assert test.text_area[0].disabled and test.file_uploader[0].disabled
    failure["permanent"] = False
    test.button(key="retry_startup").click().run(timeout=UI_TIMEOUT)
    assert not test.exception and not test.error and test.chat_input
    assert test.session_state.conversation_id == conversation and len(instances) == 1
    assert test.text_area[0].value == "Startup Example. Python and SQL experience."
    assert test.text_input[0].value == "https://www.linkedin.com/in/startup-example/"
    assert test.file_uploader[0].value.name == "startup.txt"
    monkeypatch.setenv("BUSINESS_TIMEZONE", "Fictitious/private-configuration-value")
    test.run(timeout=UI_TIMEOUT)
    assert not test.exception and "configuration needs attention" in test.error[0].value
    assert "Fictitious/private-configuration-value" not in str(test)
    monkeypatch.delenv("BUSINESS_TIMEZONE")
    test.button(key="retry_startup").click().run(timeout=UI_TIMEOUT)
    assert not test.exception and test.text_area[0].value.startswith("Startup Example.")
    assert test.file_uploader[0].value.name == "startup.txt"


def test_select_reference_job_and_start_conversation(backend):
    from app.config import Settings
    from app.ingestion.registry import JobRegistry

    backend.plan.return_value = backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    description = (
        "Responsibilities:\n• "
        + "Build Python services and maintain reliable data pipelines. " * 20
        + "\nQualifications:\n• Experience with Python, SQL and generic types such as <T>."
        "\nBenefits:\nThe complete description ends with professional development support."
    )
    registry = JobRegistry(Settings(_env_file=None))
    try:
        registry.source({"provider": "greenhouse", "board": "example", "company": "Example"})
        registry.save_job(
            {
                "job_id": "reference",
                "source_id": "greenhouse:example",
                "external_job_id": "1",
                "canonical_url": "https://example.com/1",
                "company_name": "Example",
                "title": "Python Backend Engineer",
                "description_text": description,
                "locations": [{"location_text": "London"}, {"location_text": "Unknown"}],
                "workplace_type": "hybrid",
            }
        )
    finally:
        registry.close()
    test = AppTest.from_file(str(ROOT / "streamlit_app/streamlit_main.py")).run(timeout=UI_TIMEOUT)
    assert not test.exception
    starters = [question.text for question in STARTER_QUESTIONS]
    assert [
        button.label for button in test.button if (button.key or "").startswith("suggestion_")
    ] == starters
    assert not test.chat_message
    assert not any('class="role-location"' in item.proto.body for item in test.get("html"))
    assert (
        "Python Backend Engineer · Example · London · Hybrid"
        in test.selectbox(key="selected_job_id").options
    )
    test.selectbox(key="selected_job_id").select("reference").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert [
        button.label for button in test.button if (button.key or "").startswith("suggestion_")
    ] == starters
    test.button(key="job_details").click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    rendered = next(
        item.proto.body for item in test.get("html") if 'class="job-description"' in item.proto.body
    )
    assert description.splitlines()[-1] in rendered
    assert "<h2>Responsibilities:</h2>" in rendered and "<ul>" in rendered
    assert "&lt;T&gt;" in rendered
    location = next(
        item.proto.body for item in test.get("html") if 'class="role-location"' in item.proto.body
    )
    assert "London · Hybrid" in location and "Unknown" not in location
    assert not any("Excerpt from" in item.value for item in test.caption)
    backend.plan.assert_not_called()
    test.button(key="close_description").click().run(timeout=UI_TIMEOUT)
    test.selectbox(key="selected_job_id").select("assignment-python-developer").run(
        timeout=UI_TIMEOUT
    )
    assert not any('class="job-description"' in item.proto.body for item in test.get("html"))
    assert not any('class="role-location"' in item.proto.body for item in test.get("html"))
    test.selectbox(key="selected_job_id").select("reference").run(timeout=UI_TIMEOUT)
    test.chat_input[0].set_value("Hi, I would like an interview").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.selectbox(key="selected_job_id").disabled
    assert test.title[0].value == "Python Backend Engineer"
    test.chat_input[0].set_value("Schedule an interview").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert any("reference only" in message.markdown[0].value for message in test.chat_message)

    test.chat_input[0].set_value("Please stop texting me").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert any("conversation is complete" in item.value for item in test.caption)
    assert not any((button.key or "").startswith("suggestion_") for button in test.button)


def test_suggestion_buttons_use_background_and_only_latest_answer_questions(backend):
    from app.schemas import ProfileSections

    backend.extract_profile.return_value = ProfileSections(
        name=None, experience=[], skills=["Python"], education=[], projects=[]
    )
    test = AppTest.from_file(str(ROOT / "streamlit_app/streamlit_main.py")).run(timeout=UI_TIMEOUT)
    assert len(test.container(key="workspace").chat_input) == 1
    test.text_area[0].set_value("I use Python.").run(timeout=UI_TIMEOUT)
    test.button(key="suggestion_starter_starter-skills").click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert not test.container(key="workspace").chat_input
    assert len(test.chat_input) == 1
    backend.extract_profile.assert_called_once_with("I use Python.")
    first_buttons = [
        button for button in test.button if (button.key or "").startswith("suggestion_")
    ]
    assert len(first_buttons) == 3
    first_texts = [button.label for button in first_buttons]
    assert [button.label for button in test.chat_message[-1].button] == first_texts
    assert test.chat_message[0].markdown[0].value == STARTER_QUESTIONS[0].text
    calls = backend.suggest_followups.call_count
    test.run(timeout=UI_TIMEOUT)
    assert backend.suggest_followups.call_count == calls
    first_buttons = [
        button for button in test.button if (button.key or "").startswith("suggestion_")
    ]
    chosen = first_buttons[0].label
    first_buttons[0].click().run(timeout=UI_TIMEOUT)
    assert not test.exception and len(test.chat_message) == 4
    assert test.chat_message[2].markdown[0].value == chosen
    latest = [
        button.label for button in test.button if (button.key or "").startswith("suggestion_")
    ]
    assert len(latest) == 3 and not set(first_texts) & set(latest)
    assert [button.label for button in test.chat_message[-1].button] == latest
    assert not test.chat_message[1].button


def test_review_booking_retry_and_new_conversation(backend, repository, monkeypatch):
    from datetime import UTC, datetime, timedelta
    from io import BytesIO

    from reportlab.pdfgen.canvas import Canvas
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session
    from streamlit.runtime.context import ContextProxy

    from app.schemas import ProfileSections
    from app.service import ChatService
    from app.storage.models import Booking, CandidateProfile, CandidateSource, Slot

    # All rows belong to this test's disposable PostgreSQL database.
    start = (datetime.now(UTC) + timedelta(days=2)).replace(
        hour=9, minute=0, second=0, microsecond=0, tzinfo=None
    )
    with Session(repository.engine) as session, session.begin():
        session.add_all(
            [
                Slot(
                    starts_at=start + timedelta(minutes=30 * index),
                    duration_minutes=30,
                    position="Python Dev",
                    available=True,
                )
                for index in range(3)
            ]
        )
    backend.plan.return_value = backend.plan.return_value.model_copy(
        update={"scheduling_needed": True}
    )
    original = ChatService.turn
    confirmations = []
    processed_states = {}

    def uncertain_response(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        processed_states[kwargs["conversation_id"]] = self.state(kwargs["conversation_id"])
        if kwargs.get("selected_slot_id"):
            confirmations.append(kwargs["request_id"])
            if len(confirmations) == 1:
                raise OSError("Test-only lost response after database commit")
        return result

    monkeypatch.setattr(ChatService, "turn", uncertain_response)
    browser_zone = {"value": "Asia/Jerusalem"}
    monkeypatch.setattr(ContextProxy, "timezone", property(lambda self: browser_zone["value"]))
    test = AppTest.from_file(str(ROOT / "streamlit_app/streamlit_main.py")).run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.chat_input and len(test.chat_message) == 0
    assert test.session_state.timezone == "Asia/Jerusalem"
    assert len(test.selectbox) == 1
    background = "Alex Reset Example. 3 years of Python experience. Skills: Python."
    backend.extract_profile.return_value = ProfileSections(
        name="Alex Reset Example",
        experience=["3 years of Python experience."],
        skills=["Python"],
        education=[],
        projects=[],
    )
    pdf = BytesIO()
    canvas = Canvas(pdf)
    canvas.drawString(72, 720, background)
    canvas.save()
    old_upload_key = test.file_uploader[0].key
    test.text_area[0].set_value(background).run(timeout=UI_TIMEOUT)
    test.text_input[0].set_value("https://www.linkedin.com/in/reset-example/").run(
        timeout=UI_TIMEOUT
    )
    test.file_uploader[0].upload("first-candidate.pdf", pdf.getvalue(), "application/pdf").run(
        timeout=UI_TIMEOUT
    )
    assert test.file_uploader[0].value.name == "first-candidate.pdf"
    old_upload = test.file_uploader[0].value
    next(
        button for button in test.button if button.label == "How can I arrange an interview?"
    ).click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert any(item.value == "Times shown in Asia/Jerusalem" for item in test.caption)
    assert "Available appointments:" not in test.chat_message[-1].markdown[0].value
    assert (
        len([button for button in test.chat_message[-1].button if button.label == "Choose time"])
        == 3
    )
    # A browser timezone change must not alter an active booking conversation.
    browser_zone["value"] = "America/New_York"
    test.run(timeout=UI_TIMEOUT)
    assert test.session_state.timezone == "Asia/Jerusalem"
    choices = [button for button in test.button if button.label == "Choose time"]
    assert len(choices) == 3
    choices[0].click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert any("30 minutes · Asia/Jerusalem (UTC" in item.value for item in test.caption)
    with Session(repository.engine) as session:
        assert session.scalar(select(func.count()).select_from(Booking)) == 0
    next(button for button in test.button if button.label == "Choose another time").click().run(
        timeout=UI_TIMEOUT
    )
    assert not confirmations
    next(button for button in test.button if button.label == "Choose time").click().run(
        timeout=UI_TIMEOUT
    )
    test.button(key="confirm_interview").click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.error and test.chat_input[0].disabled
    # A display rerun must not retry a submission with an uncertain outcome.
    test.run(timeout=UI_TIMEOUT)
    assert len(confirmations) == 1
    # Startup recovery must not resubmit a committed request whose response was
    # lost. Keep both durable IDs, the failure flag, and the candidate's inputs.
    from psycopg.errors import AdminShutdown

    original_state = ChatService.state
    pending = dict(test.session_state.pending_request)
    pending_conversation = test.session_state.conversation_id
    failures = {"remaining": 3}

    def temporarily_disconnected(self, conversation_id):
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise AdminShutdown("Fictitious startup outage")
        return original_state(self, conversation_id)

    monkeypatch.setattr(ChatService, "state", temporarily_disconnected)
    test.run(timeout=UI_TIMEOUT)
    assert not test.exception and test.error
    assert test.button(key="retry_startup")
    assert test.session_state.pending_request == pending
    assert test.session_state.conversation_id == pending_conversation
    assert len(confirmations) == 1
    test.button(key="retry_startup").click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.session_state.pending_request == pending
    assert test.session_state.conversation_id == pending_conversation
    assert test.session_state.request_failed
    assert len(confirmations) == 1
    next(button for button in test.button if button.label == "Retry request").click().run(
        timeout=UI_TIMEOUT
    )
    assert not test.exception
    assert len(confirmations) == 2 and confirmations[0] == confirmations[1]
    assert any(item.value == "Interview confirmed" for item in test.success)
    old_conversation = test.session_state.conversation_id
    next(button for button in test.button if button.label == "Start new conversation").click().run(
        timeout=UI_TIMEOUT
    )
    assert not test.exception
    assert test.session_state.conversation_id != old_conversation
    assert test.session_state.timezone == "America/New_York"
    assert not test.selectbox(key="selected_job_id").disabled
    assert len(test.chat_message) == 0 and test.chat_input
    assert test.file_uploader[0].key != old_upload_key
    assert test.file_uploader[0].value is None
    assert test.text_area[0].value == test.text_input[0].value == ""
    assert not any("Background used:" in item.value for item in test.caption)
    # A late browser update can restore an old widget value, but neither that
    # value nor a stale queued upload may become the next candidate's context.
    new_conversation = test.session_state.conversation_id
    test.session_state[old_upload_key] = old_upload
    test.session_state["pending_request"] = {
        "conversation_id": old_conversation,
        "request_id": "stale-upload",
        "text": "Previous candidate",
        "context_inputs": {
            "introduction": background,
            "file_name": old_upload.name,
            "file_data": old_upload.getvalue(),
        },
    }
    test.session_state["attempt_request"] = True
    test.session_state["pending_answer"] = {
        "request_id": "stale-upload",
        "result": {"response": background},
    }
    calls = backend.plan.call_count
    test.run(timeout=UI_TIMEOUT)
    assert not test.exception and backend.plan.call_count == calls
    assert "pending_request" not in test.session_state
    assert "pending_answer" not in test.session_state
    assert test.file_uploader[0].value is None
    assert test.text_area[0].value == test.text_input[0].value == ""
    assert not test.chat_message and test.session_state.conversation_id == new_conversation
    backend.plan.return_value = backend.plan.return_value.model_copy(
        update={
            "scheduling_needed": False,
            "info_question": "What do you know about my background?",
        }
    )
    backend.exit_label.return_value = "end"
    extraction_calls = backend.extract_profile.call_count
    test.chat_input[0].set_value("What do you know about my background?").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert backend.extract_profile.call_count == extraction_calls
    context = processed_states[new_conversation]
    assert not context["profile_source_ids"]
    assert not any(context.get("candidate_profile", {}).get("sections", {}).values())
    assert "Alex Reset Example" not in str(context)
    assert "info" not in context["advisors"]
    assert (
        test.chat_message[-1].markdown[0].value
        == "Could you share your name and Python experience, or upload a CV?"
    )
    backend.plan.return_value = backend.plan.return_value.model_copy(
        update={"info_question": "What responsibilities does this role have?"}
    )
    test.chat_input[0].set_value(
        "How could my experience fit this role, and what responsibilities does it have?"
    ).run(timeout=UI_TIMEOUT)
    assert not test.exception and "info" in processed_states[new_conversation]["advisors"]
    with Session(repository.engine) as session:
        assert session.scalar(select(func.count()).select_from(Booking)) == 1
        assert (
            session.scalar(
                select(func.count())
                .select_from(CandidateSource)
                .where(CandidateSource.conversation_id == new_conversation)
            )
            == 0
        )
        sources = list(
            session.scalars(
                select(CandidateSource).where(CandidateSource.conversation_id == old_conversation)
            )
        )
        assert len(sources) == 3
        assert any(row.filename == "first-candidate.pdf" for row in sources)
        assert session.scalar(
            select(CandidateProfile).where(
                CandidateProfile.conversation_id == old_conversation,
                CandidateProfile.is_current.is_(True),
            )
        ).data["sections"]["name"] == ["Alex Reset Example"]
