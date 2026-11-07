from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

from app.guardrails import booking_allowed
from app.ingestion.models import DEFAULT_JOB_ID
from app.observability.logging import log_event
from app.observability.tracing import traced
from app.schemas import ScheduleRequest
from app.storage.schedule_repository import BookingConflict, ScheduleRepository
from app.tools.dates import local_instant, parse_date, parse_time


def scheduling_tool(repository: ScheduleRepository, state: dict):
    @traced("tool", "manage_interview")
    def manage_interview(**kwargs) -> dict:
        """Search SQL availability, or book a previously offered slot explicitly accepted by the candidate."""
        request = ScheduleRequest(**kwargs)
        now = datetime.fromisoformat(state["reference_time"])
        zone = state["timezone"]
        offered = state.get("offered_slots", [])
        job_id = state.get("job_id", DEFAULT_JOB_ID)
        future = now + timedelta(microseconds=1)

        def alternatives(anchor, message, *, excluded=(), reason):
            slots = repository.nearest(
                future,
                zone,
                anchor=anchor,
                job_id=job_id,
                exclude_ids=sorted(set(state.get("declined_slot_ids", [])) | set(excluded)),
            )
            log_event(
                "calendar_alternatives_offered",
                component="app.calendar",
                job_id=job_id,
                timezone=zone,
                slot_count=len(slots),
                reason_code=reason,
            )
            return {
                "status": "alternatives" if slots else "unavailable",
                "slots": slots,
                "message": message
                + (
                    " These are the nearest available alternatives. Please choose one to confirm."
                    if slots
                    else " No other appointments are currently available; a recruiter needs to add availability."
                ),
            }

        if not state.get("scheduling_enabled", job_id == DEFAULT_JOB_ID):
            return {
                "status": "unavailable",
                "message": "This job is available for reference only. Interview availability is not configured; apply through the original employer posting.",
                "slots": [],
            }
        existing = repository.get_booking(state["conversation_id"], zone, job_id)
        if existing:
            return {"status": "already_booked", "booking": existing, "slots": []}
        try:
            day = parse_date(request.date_text, now, zone)
            at = parse_time(request.time_text)
            if day and at:
                local_instant(day, at, zone)  # Reject ambiguous/nonexistent DST times.
            selected = state.get("selected_slot_id") or request.selected_slot_id
            accepted = bool(state.get("selected_slot_id")) or request.accepts_slot
            if accepted and not selected and (day or at):
                matches = []
                for item in offered:
                    local = datetime.fromisoformat(item["starts_at"]).astimezone(ZoneInfo(zone))
                    if (day is None or local.date() == day) and (
                        at is None or local.time().replace(tzinfo=None) == at
                    ):
                        matches.append(item)
                if len(matches) == 1:
                    selected = matches[0]["id"]
            if selected:
                if not accepted or selected not in {item["id"] for item in offered}:
                    log_event(
                        "calendar_confirmation_rejected",
                        component="app.calendar",
                        job_id=job_id,
                        reason_code="unoffered_or_unaccepted_slot",
                    )
                    return {
                        "status": "clarify",
                        "message": "Please choose one of the offered appointments.",
                        "slots": offered,
                    }
                if not booking_allowed(state, selected, day=day, at=at, request=request):
                    log_event(
                        "calendar_confirmation_rejected",
                        component="app.calendar",
                        job_id=job_id,
                        slot_id=selected,
                        reason_code="consent_required",
                    )
                    return {
                        "status": "clarify",
                        "message": "Please confirm which offered appointment you want to book.",
                        "slots": offered,
                    }
                try:
                    booking = repository.book(
                        selected,
                        state["conversation_id"],
                        state["request_id"],
                        now,
                        zone,
                        job_id=job_id,
                    )
                except BookingConflict:
                    existing = repository.get_booking(state["conversation_id"], zone, job_id)
                    if existing:
                        return {"status": "already_booked", "booking": existing, "slots": []}
                    anchor = datetime.fromisoformat(
                        next(slot["starts_at"] for slot in offered if slot["id"] == selected)
                    )
                    return alternatives(
                        anchor,
                        "That appointment is no longer available.",
                        excluded=(selected,),
                        reason="booking_conflict",
                    )
                return {"status": "booked", "booking": booking, "slots": []}
            if accepted and not day and not at and offered:
                return {
                    "status": "clarify",
                    "message": "Which appointment would you like to confirm?",
                    "slots": offered,
                }
            if day:
                start = max(future, local_instant(day, time.min, zone))
                end = local_instant(day + timedelta(days=1), time.min, zone)
                if end <= now:
                    return {
                        "status": "clarify",
                        "message": "That date has passed. Which future date works for you?",
                        "slots": [],
                    }
            else:
                start, end = future, None
            slots = repository.available(
                start,
                end,
                zone,
                job_id=job_id,
                exact_time=at,
                period=request.period,
                exclude_ids=state.get("declined_slot_ids", []),
            )
            if not slots and (day or at or request.period):
                anchor = local_instant(
                    day or now.astimezone(ZoneInfo(zone)).date(), at or time.min, zone
                )
                return alternatives(
                    anchor,
                    "No appointments match your requested date or time.",
                    reason="preferences_unavailable",
                )
            return {
                "status": "offered" if slots else "unavailable",
                "slots": slots,
                "message": ""
                if slots
                else "No future appointments are currently available; a recruiter needs to add availability.",
            }
        except ValueError as exc:
            log_event(
                "calendar_request_rejected",
                component="app.calendar",
                job_id=job_id,
                reason_code="invalid_preferences",
                error_code=type(exc).__name__,
            )
            return {"status": "clarify", "message": str(exc), "slots": []}

    return StructuredTool.from_function(
        manage_interview, name="manage_interview", args_schema=ScheduleRequest
    )
