import hashlib
import html
import logging
import re
import sys
from base64 import b64encode
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo, available_timezones

# Streamlit executes this file directly; anchor imports to the repository, not cwd.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.config import Settings
from app.ingestion.description import description_html
from app.ingestion.models import DEFAULT_JOB_ID
from app.observability.logging import log_event
from app.service import ChatService
from app.suggestions import STARTER_QUESTIONS
from streamlit_app.introduction import (
    clear_introduction_inputs,
    context_snapshot,
    render_introduction,
    render_profile_status,
)
from streamlit_app.page_actions import render_page_actions
from streamlit_app.presentation import response_text
from streamlit_app.ui_behaviors import render_ui_behaviors

st.set_page_config(page_title="Recruitment assistant", page_icon=":material/forum:", layout="wide")


def load_settings():
    values = {}
    try:
        allowed = set(Settings.model_fields)
        values = {
            key.lower(): value
            for key, value in st.secrets.to_dict().items()
            if key.lower() in allowed
        }
    except FileNotFoundError:
        pass
    return Settings(**values)


@st.cache_resource
def create_service(_settings, configuration_identity):
    """Cache the advisor-loop service for this application configuration."""
    return ChatService(_settings)


def queue_request(**request):
    if not st.session_state.get("pending_request"):
        st.session_state.pending_request = {
            "request_id": str(uuid4()),
            "context_inputs": context_snapshot(),
            **request,
        }
        st.session_state.attempt_request = True
        st.session_state.pop("request_failed", None)
        st.session_state.pop("review_slot_id", None)


def submit_message():
    message = st.session_state.get("candidate_message", "")
    if message and message.strip():
        queue_request(text=message.strip())


def reset_conversation():
    # Recorded history and confirmed bookings remain in PostgreSQL.
    st.session_state.conversation_id = str(uuid4())
    for key in (
        "pending_request",
        "attempt_request",
        "request_failed",
        "review_slot_id",
        "candidate_message",
        "pending_answer",
    ):
        st.session_state.pop(key, None)
    clear_introduction_inputs()


def select_slot(slot_id):
    st.session_state.review_slot_id = slot_id


def clear_slot():
    st.session_state.pop("review_slot_id", None)


def retry_request():
    st.session_state.attempt_request = True


def initialize_timezone(settings, state):
    if state.get("timezone"):
        zone = state["timezone"]
    elif st.session_state.get("pending_request"):
        # Keep retries in the same timezone as the original submission.
        zone = st.session_state.get("timezone", settings.business_timezone)
    else:
        browser_zone = st.context.timezone
        zone = browser_zone if browser_zone in available_timezones() else settings.business_timezone
    st.session_state.timezone = zone


def employer(job):
    return "" if job["job_id"] == DEFAULT_JOB_ID else job["company_name"]


def location_label(job):
    locations = []
    seen = set()
    for value in job.get("locations", []):
        text = " ".join((value or "").split())
        key = text.casefold().strip(" .")
        if (
            key not in {"", "unknown", "not specified", "unspecified", "n/a", "none", "null"}
            and key not in seen
        ):
            locations.append(text)
            seen.add(key)
    location = " / ".join(locations)
    mode = re.sub(r"[-_\s]", "", job.get("workplace_type") or "").casefold()
    workplace = {"remote": "Remote", "hybrid": "Hybrid", "onsite": "On-site"}.get(mode)
    pattern = "on[- ]?site" if mode == "onsite" else mode
    if workplace and not re.search(rf"\b{pattern}\b", location, re.IGNORECASE):
        location = " · ".join(filter(None, [location, workplace]))
    return location


def job_label(job):
    return " · ".join(filter(None, [job["title"], employer(job), location_label(job)]))


def public_url(value):
    return value if value and urlsplit(value).scheme in {"https", "http"} else None


def render_company_icon(job, logo):
    if not logo or logo.get("mime_type") != "image/png" or not logo.get("data"):
        return
    name = employer(job) or job["title"]
    data = b64encode(logo["data"]).decode("ascii")
    content = (
        f'<img src="data:image/png;base64,{data}" alt="{html.escape(name)} logo" '
        'width="48" height="48">'
    )
    with st.container(width=48):
        st.html(f'<div class="company-icon">{content}</div>')


def job_downloads(service, settings, job):
    downloads = []
    for document in service.registry.documents(job["job_id"], collection=settings.index_collection):
        path = Path(document["path"]).resolve()
        data = document.get("data")
        if (
            path.parent == settings.job_pdf_path.resolve()
            and path.suffix.lower() == ".pdf"
            and path.is_file()
        ):
            try:
                if data is not None:
                    downloads.append({"filename": path.name, "data": data})
                    continue
                data = path.read_bytes()
            except OSError:
                pass
        downloads.append({"filename": path.name, "data": data})
    return downloads


def render_header(job, choices, locked, started, logo, downloads):
    with st.container(key="role_header", horizontal=True, vertical_alignment="top"):
        with st.container():
            with st.container(key="role_title", horizontal=True, vertical_alignment="top"):
                render_company_icon(job, logo)
                with st.container():
                    st.title(job["title"], anchor=False)
                    company, location = employer(job), location_label(job)
                    if company or location:
                        with st.container(gap="xxsmall"):
                            if company:
                                st.write(company)
                            if location:
                                icon = (
                                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                                    'width="16" height="16" aria-hidden="true" '
                                    'fill="none" stroke="currentColor" stroke-width="1.75" '
                                    'stroke-linecap="round" stroke-linejoin="round">'
                                    '<path d="M20 10c0 6-8 12-8 12S4 16 4 10a8 8 0 1 1 16 0Z"/>'
                                    '<circle cx="12" cy="10" r="3"/></svg>'
                                )
                                st.html(
                                    '<p class="role-location">'
                                    f"{icon}"
                                    f"<span>{html.escape(location)}</span></p>"
                                )
        with st.container(width="content", horizontal=True, gap="small", key="header_controls"):
            render_page_actions(downloads)
            with st.popover(
                "Change role",
                icon=":material/swap_horiz:",
                disabled=locked,
                help="Start a new conversation to change roles." if locked else None,
            ):
                st.selectbox(
                    "Position",
                    list(choices),
                    format_func=lambda value: job_label(choices[value]),
                    key="selected_job_id",
                    label_visibility="collapsed",
                    disabled=locked,
                )
            if started:
                with st.popover(
                    "New conversation",
                    icon=":material/add:",
                    disabled=bool(st.session_state.get("pending_request")),
                ):
                    st.write("Start a fresh conversation for this or another position.")
                    st.caption("Your recorded messages and any confirmed interview stay saved.")
                    st.button("Start new conversation", on_click=reset_conversation)
    st.caption(
        "Interview booking available"
        if job["scheduling_enabled"]
        else "Reference posting · Interview requests go through the employer"
    )


@st.dialog("Job description", width="large", on_dismiss="rerun")
def show_job_description(job):
    with st.container(key="job_description_reader"):
        st.subheader(job["title"], anchor=False)
        description = job.get("description_text", "").strip()
        if description:
            st.html(
                description_html(
                    description, title=job["title"], format=job.get("description_format", "plain")
                )
            )
        else:
            st.write(
                "Read the published job description for responsibilities, skills and experience."
            )
        if url := public_url(job["canonical_url"]):
            st.link_button("View original employer posting", url, icon=":material/open_in_new:")
        for notice in job.get("description_notices", []):
            st.caption(notice)
        if st.button("Close description", key="close_description"):
            st.rerun()


def render_job_details(job):
    if st.button("About this role", key="job_details", icon=":material/work_outline:"):
        show_job_description(job)


def render_suggestions(questions, busy, answer_request_id=None):
    if busy or not questions:
        return
    with st.container(horizontal=True, key="suggestions"):
        for question in questions:
            st.button(
                question["text"],
                key=f"suggestion_{answer_request_id or 'starter'}_{question['id']}",
                width="stretch",
                on_click=queue_request,
                kwargs={
                    "text": question["text"],
                    "suggestion": {
                        "suggestion_id": question["id"],
                        "answer_request_id": answer_request_id,
                    },
                },
            )


def render_sources(source_ids, references, job, index):
    if not source_ids:
        return
    pages = sorted(
        {int(row["page"]) for key in source_ids if (row := references.get(key)) and row.get("page")}
    )
    if pages:
        label = (
            "Source · " + ("page " if len(pages) == 1 else "pages ") + ", ".join(map(str, pages))
        )
        st.caption(label)
    with st.expander("Sources", icon=":material/description:", key=f"sources_{index}"):
        groups = {}
        for source_id in source_ids:
            if reference := references.get(source_id):
                name = Path(reference.get("source", "Job description.pdf")).name
                pages = groups.setdefault(name, set())
                if page := reference.get("page"):
                    pages.add(int(page))
        for name, pages in groups.items():
            suffix = ", ".join(str(page) for page in sorted(pages))
            label = ("page " if len(pages) == 1 else "pages ") + suffix
            st.caption(f"{name} — {label}" if pages else name)
        if not groups or any(key not in references for key in source_ids):
            st.caption("Some earlier source details are unavailable. Read the current PDF above.")
        if url := public_url(job["canonical_url"]):
            st.link_button("Original employer posting", url, key=f"source_link_{index}")


def render_messages(service, messages, job, busy, closed):
    source_ids = list(
        dict.fromkeys(
            key for message in messages for key in message["result"].get("source_ids", [])
        )
    )
    try:
        references = {
            row["id"]: row for row in service.retriever.references(source_ids, job["job_id"])
        }
    except Exception as error:
        log_event(
            "source_details_unavailable",
            level=logging.WARNING,
            component="app.streamlit",
            error_code=type(error).__name__,
        )
        references = {}
    for index, message in enumerate(messages):
        assistant = message["role"] == "assistant"
        latest = index == len(messages) - 1
        with st.chat_message(
            message["role"], avatar=":material/forum:" if assistant else ":material/person:"
        ):
            result = message["result"]
            cards = (
                assistant
                and latest
                and bool(
                    result.get("booking")
                    or (job["scheduling_enabled"] and result.get("offered_slots"))
                )
            )
            if cards and result.get("booking"):
                render_appointments(result, job, busy)
            text = response_text(message["content"], result, cards=cards)
            if text:
                st.markdown(text)
            if assistant:
                if cards and not result.get("booking"):
                    render_appointments(result, job, busy or closed)
                render_sources(message["result"].get("source_ids", []), references, job, index)
                if not closed and latest:
                    render_suggestions(
                        message["result"].get("suggested_questions", []),
                        busy,
                        message["request_id"],
                    )
    pending = st.session_state.get("pending_request")
    if pending and not any(row["request_id"] == pending["request_id"] for row in messages):
        text = pending.get("text")
        if text:
            with st.chat_message("user", avatar=":material/person:"):
                st.write(text)


def slot_details(slot):
    zone = ZoneInfo(st.session_state.timezone)
    start = datetime.fromisoformat(slot["starts_at"]).astimezone(zone)
    end = datetime.fromisoformat(slot["ends_at"]).astimezone(zone) if slot.get("ends_at") else None
    date = start.strftime("%A, %d %B %Y")
    times = start.strftime("%H:%M")
    if end:
        times += "–" + end.strftime("%H:%M")
    duration = f"{slot['duration_minutes']} minutes" if slot.get("duration_minutes") else ""
    offset = start.strftime("%z")
    offset = f"UTC{offset[:3]}:{offset[3:]}"
    return date, times, " · ".join(filter(None, [duration, f"{zone.key} ({offset})"]))


def render_appointments(result, job, busy):
    if booking := result.get("booking"):
        date, times, detail = slot_details(booking["slot"])
        with st.container(border=True, key="booking_receipt"):
            st.success("Interview confirmed", icon=":material/check_circle:")
            st.write(f"**{date} · {times}**")
            st.caption(detail)
            st.caption(
                "Your appointment is recorded. Calendar invitations are not sent by this app."
            )
        return
    slots = result.get("offered_slots", [])[:3]
    if not slots or not job["scheduling_enabled"]:
        return
    review = next(
        (slot for slot in slots if slot["id"] == st.session_state.get("review_slot_id")), None
    )
    if review:
        date, times, detail = slot_details(review)
        with st.container(border=True, key="appointment_review"):
            st.subheader("Review your interview", anchor=False)
            st.write(job_label(job))
            st.write(f"**{date} · {times}**")
            st.caption(detail)
            st.caption("Your appointment is not booked until you confirm.")
            with st.container(horizontal=True):
                st.button(
                    "Confirm interview",
                    type="primary",
                    key="confirm_interview",
                    on_click=queue_request,
                    kwargs={
                        "text": f"Please confirm {review['label']}.",
                        "selected_slot_id": review["id"],
                    },
                    disabled=busy,
                )
                st.button("Choose another time", on_click=clear_slot, disabled=busy)
        return
    st.subheader("Choose an interview time", anchor=False)
    st.caption(f"Times shown in {st.session_state.timezone}")
    with st.container(horizontal=True, gap="small", key="appointment_choices"):
        for slot in slots:
            date, times, detail = slot_details(slot)
            with st.container(border=True, key=f"slot_card_{slot['id']}"):
                if slot.get("is_alternative"):
                    st.caption("Nearby alternative")
                st.write(f"**{date}**")
                st.subheader(times, anchor=False)
                st.caption(detail)
                st.button(
                    "Choose time",
                    key=f"slot_{slot['id']}",
                    width="stretch",
                    on_click=select_slot,
                    args=(slot["id"],),
                    disabled=busy,
                )
    st.caption("Choose a time to review it, or suggest another date in the conversation.")


def process_pending(service, selected_job, messages):
    pending = st.session_state.get("pending_request")
    if not pending:
        return
    attempt = st.session_state.pop("attempt_request", False)
    saved = any(
        row["role"] == "assistant" and row["request_id"] == pending["request_id"]
        for row in messages
    )
    with st.chat_message("assistant", avatar=":material/forum:"):
        answer = st.empty()
        status_holder = st.empty()
        if previous := st.session_state.get("pending_answer"):
            if previous["request_id"] == pending["request_id"] and not saved:
                answer.markdown(previous["result"]["response"])
        if attempt:
            status = (
                "Confirming your interview…"
                if pending.get("selected_slot_id")
                else "Reading your background and preparing a reply…"
                if any(pending.get("context_inputs", {}).values())
                else "Preparing a reply…"
            )
            try:
                progress = status_holder.status(status, expanded=False)

                def report(event):
                    if event.get("stage") == "answer_ready":
                        st.session_state.pending_answer = event
                        answer.markdown(event["result"]["response"])
                        progress.update(label=event["label"])
                    elif event.get("label"):
                        progress.update(label=event["label"])

                service.turn(
                    st.session_state.conversation_id,
                    timezone=st.session_state.timezone,
                    job_id=selected_job,
                    progress=report,
                    **pending,
                )
                for key in (
                    "pending_request",
                    "request_failed",
                    "review_slot_id",
                    "pending_answer",
                ):
                    st.session_state.pop(key, None)
                st.rerun()
            except Exception:
                st.session_state.request_failed = True
        if st.session_state.get("request_failed"):
            status_holder.empty()
            st.error(
                "The response was interrupted. Your submitted information is kept; retry to check the result."
            )
            if pending.get("selected_slot_id"):
                st.caption("Retry checks the same request, so it will not book twice.")
            st.button("Retry request", on_click=retry_request, type="primary")


def render_composer(closed, busy, started):
    target = st.bottom if started else st
    with target.container(horizontal=True, horizontal_alignment="center"):
        with st.container(width=960, key="composer"):
            if closed:
                st.caption("This conversation is complete. Use New conversation to start again.")
            else:
                if not started:
                    render_suggestions(
                        [question.model_dump() for question in STARTER_QUESTIONS], busy
                    )
                st.chat_input(
                    "Message the assistant…",
                    key="candidate_message",
                    max_chars=4000,
                    disabled=busy,
                    submit_mode="disable",
                    on_submit=submit_message,
                )
                st.caption("Messages are saved in this app")


def render():
    st.html(Path(__file__).with_name("styles.css"))
    settings = load_settings()
    if not settings.openai_api_key.get_secret_value().strip():
        st.error("Add OPENAI_API_KEY to .env or Streamlit secrets before starting the assistant.")
        st.stop()
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = str(uuid4())
    identity = hashlib.sha256(
        settings.model_dump_json().encode()
        + settings.openai_api_key.get_secret_value().encode()
        + settings.hf_token.get_secret_value().encode()
    ).hexdigest()
    try:
        service = create_service(settings, identity)
        state = service.state(st.session_state.conversation_id)
        messages = service.conversations.messages(st.session_state.conversation_id)
        jobs = sorted(
            service.registry.list_jobs(collection=settings.index_collection),
            key=lambda job: (job["job_id"] != DEFAULT_JOB_ID, job["company_name"], job["title"]),
        )
    except Exception as error:
        log_event(
            "service_initialization_failed",
            level=logging.ERROR,
            component="app.streamlit",
            error_code=type(error).__name__,
        )
        st.error("The assistant is temporarily unavailable. Please try again shortly.")
        st.stop()
    choices = {job["job_id"]: job for job in jobs}
    if not choices:
        st.info("No active positions are available. Please try again later.")
        st.stop()
    conversation_job = state.get("job_id")
    if conversation_job and conversation_job not in choices:
        st.info("This position is no longer active. Start a new conversation for another role.")
        st.button("Choose another position", on_click=reset_conversation)
        st.stop()
    if conversation_job:
        st.session_state.selected_job_id = conversation_job
    elif st.session_state.get("selected_job_id") not in choices:
        st.session_state.selected_job_id = next(iter(choices))
    initialize_timezone(settings, state)
    selected_job = st.session_state.selected_job_id
    job = service.registry.get_job(selected_job, collection=settings.index_collection)
    logo = service.registry.company_logo(selected_job)
    downloads = job_downloads(service, settings, job)
    busy = bool(st.session_state.get("pending_request"))
    started = bool(state or messages)
    closed = state.get("conversation_status") == "closed" and not busy
    with st.container(horizontal=True, horizontal_alignment="center"):
        with st.container(width=960, key="workspace"):
            render_header(
                job,
                choices,
                locked=started or busy,
                started=started,
                logo=logo,
                downloads=downloads,
            )
            render_job_details(job)
            st.divider()
            if not closed:
                render_introduction(busy, state.get("candidate_profile", {}))
            render_profile_status(state)
            render_messages(service, messages, job, busy, closed)
            if not started:
                render_composer(closed, busy, started=False)
            latest_id = next(
                (row["request_id"] for row in reversed(messages) if row["role"] == "assistant"),
                None,
            )
            render_ui_behaviors(st.session_state.conversation_id, latest_id)
            process_pending(service, selected_job, messages)
    if started:
        render_composer(closed, busy, started=True)


render()
