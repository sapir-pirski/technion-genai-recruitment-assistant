from streamlit.testing.v1 import AppTest

from app.config import ROOT
from app.suggestions import STARTER_QUESTIONS

# These checks assert UI behavior against hosted PostgreSQL, not a latency target.
UI_TIMEOUT = 60


def test_only_assignment_job_is_shown_with_legacy_postings_in_storage(backend, legacy_job):
    test = AppTest.from_file(str(ROOT / "streamlit_app/streamlit_main.py")).run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.title[0].value == "Python Developer"
    assert not test.selectbox
    assert not any(item.label == "Change role" for item in test.get("popover"))
    assert [
        button.label for button in test.button if (button.key or "").startswith("suggestion_")
    ] == [question.text for question in STARTER_QUESTIONS]
    assert not test.chat_message
    backend.plan.assert_not_called()
    test.button(key="job_details").click().run(timeout=UI_TIMEOUT)
    assert not test.exception
    test.button(key="close_description").click().run(timeout=UI_TIMEOUT)
    test.chat_input[0].set_value("Hello, I use Python").run(timeout=UI_TIMEOUT)
    assert not test.exception
    assert test.title[0].value == "Python Developer" and not test.selectbox
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
    test.text_area(key="introduction_text").set_value("I use Python.").run(timeout=UI_TIMEOUT)
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

    from sqlalchemy import func, select
    from sqlalchemy.orm import Session
    from streamlit.runtime.context import ContextProxy

    from app.service import ChatService
    from app.storage.models import Booking, Slot

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

    def uncertain_response(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
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
    assert not test.selectbox
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
    assert not test.selectbox
    assert len(test.chat_message) == 0 and test.chat_input
    with Session(repository.engine) as session:
        assert session.scalar(select(func.count()).select_from(Booking)) == 1
