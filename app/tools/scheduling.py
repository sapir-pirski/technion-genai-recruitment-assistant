from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from langchain_core.tools import StructuredTool

from app.agents.backends.common import latest_text
from app.guardrails import booking_allowed, candidate_accepts
from app.ingestion.models import DEFAULT_JOB_ID
from app.observability.logging import log_event
from app.observability.tracing import traced
from app.schemas import ScheduleRequest
from app.storage.schedule_repository import BookingConflict, ScheduleRepository
from app.tools.dates import local_instant
from app.tools.preferences import resolve_preferences


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
        preferences = state.get("schedule_preferences", {})
        day = at = period = None

        def nearest(anchor, *, excluded=(), limit=3):
            required = set(preferences.get("required_fields", []))
            start = future
            end = None
            if "date" in required and day:
                start = max(start, local_instant(day, time.min, zone))
                end = local_instant(day + timedelta(days=1), time.min, zone)
            return repository.nearest(
                start,
                zone,
                anchor=anchor,
                job_id=job_id,
                exclude_ids=sorted(set(state.get("declined_slot_ids", [])) | set(excluded)),
                end=end,
                exact_time=at if "time" in required else None,
                period=period if "period" in required else None,
                limit=limit,
            )

        def alternatives(anchor, message, *, excluded=(), reason):
            slots = [
                {**slot, "is_alternative": True} for slot in nearest(anchor, excluded=excluded)
            ]
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
                "preferences": preferences,
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
            day, at, period, preferences = resolve_preferences(request, state, now, zone)
            if day and at:
                local_instant(day, at, zone)  # Reject ambiguous/nonexistent DST times.
            selected = state.get("selected_slot_id") or request.selected_slot_id
            accepted = bool(state.get("selected_slot_id")) or request.accepts_slot
            if (
                not state.get("selected_slot_id")
                and accepted
                and not candidate_accepts(latest_text(state))
            ):
                # A date preference such as 'only tomorrow' is not consent to book.
                # Continue the preference search instead of preserving stale choices.
                selected, accepted = None, False
                log_event(
                    "calendar_confirmation_rejected",
                    component="app.calendar",
                    job_id=job_id,
                    reason_code="candidate_acceptance_not_stated",
                )
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
                        "preferences": preferences,
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
                        "preferences": preferences,
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
                return {
                    "status": "booked",
                    "booking": booking,
                    "slots": [],
                    "preferences": preferences,
                }
            if accepted and not day and not at and offered:
                return {
                    "status": "clarify",
                    "message": "Which appointment would you like to confirm?",
                    "slots": offered,
                    "preferences": preferences,
                }
            if day:
                start = max(future, local_instant(day, time.min, zone))
                end = local_instant(day + timedelta(days=1), time.min, zone)
                if end <= now:
                    return {
                        "status": "clarify",
                        "message": "That date has passed. Which future date works for you?",
                        "slots": [],
                        "preferences": preferences,
                    }
            else:
                start, end = future, None
            slots = repository.available(
                start,
                end,
                zone,
                job_id=job_id,
                exact_time=at,
                period=period,
                exclude_ids=state.get("declined_slot_ids", []),
            )
            anchor = local_instant(
                day or now.astimezone(ZoneInfo(zone)).date(),
                at
                or (
                    time(15)
                    if period == "afternoon"
                    else time(9)
                    if period == "morning"
                    else time.min
                ),
                zone,
            )
            if not slots and (day or at or period):
                return alternatives(
                    anchor,
                    "No appointments match your requested date or time.",
                    reason="preferences_unavailable",
                )
            extras = []
            if 0 < len(slots) < 3 and (day or at or period):
                extras = nearest(
                    anchor, excluded=[slot["id"] for slot in slots], limit=3 - len(slots)
                )
            slots = [{**slot, "is_alternative": False} for slot in slots] + [
                {**slot, "is_alternative": True} for slot in extras
            ]
            message = (
                "Some nearby alternatives are also shown because fewer than three appointments match your preferences."
                if extras
                else ""
            )
            if slots and len(slots) < 3:
                message = (
                    "Only these appointments are currently available within your stated limits."
                )
            return {
                "status": "offered" if slots else "unavailable",
                "slots": slots,
                "preferences": preferences,
                "message": message
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
            return {"status": "clarify", "message": str(exc), "slots": offered}

    return StructuredTool.from_function(
        manage_interview, name="manage_interview", args_schema=ScheduleRequest
    )
