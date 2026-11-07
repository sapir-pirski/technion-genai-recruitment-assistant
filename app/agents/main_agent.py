import hashlib
import json

from app.agents.backends.common import explicit_stop, latest_text
from app.schemas import Plan, TurnResult


def fallback_plan(state=None):
    state = state or {}
    job = state.get("job_context", {})
    role = f"{job['title']} at {job['company_name']}" if job else "the Python Developer role"
    return Plan(
        next_advisor="respond",
        stop_requested=False,
        closure_reason=None,
        declined_slot_ids=[],
        info_question=None,
        scheduling_needed=False,
        next_question=f"Thanks for your interest in {role}. What Python projects have you worked on recently?",
    )


def task_fingerprint(state, plan):
    value = {
        "advisor": plan["next_advisor"],
        "history": state.get("messages", []),
        "job_id": state.get("job_id"),
        "reference_time": state.get("reference_time"),
        "timezone": state.get("timezone"),
        "question": plan.get("info_question") if plan["next_advisor"] == "info" else None,
        "selected_slot_id": state.get("selected_slot_id")
        if plan["next_advisor"] == "scheduling"
        else None,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def pending_advisor(state, intent):
    consulted = {result["advisor"] for result in state.get("advisor_results", [])}
    if intent.get("info_question") and "info" not in consulted:
        return "info"
    if intent.get("scheduling_needed") and "scheduling" not in consulted:
        return "scheduling"
    return None


def make_plan(backend, state: dict) -> dict:
    count = state.get("consultation_count", 0)
    consulted = {result["advisor"] for result in state.get("advisor_results", [])}
    previous = {
        key: value
        for key, value in (state.get("plan") or fallback_plan(state).model_dump()).items()
        if key in Plan.model_fields
    }
    previous.setdefault("closure_reason", None)  # Read older checkpoints without rewriting them.
    reason = None
    rerouted_from = None
    if state.get("error"):
        plan = Plan(**{**previous, "next_advisor": "respond"})
        reason = "advisor_error"
    elif count >= state.get("max_consultations", 3):
        plan = Plan(**{**previous, "next_advisor": "respond"})
        reason = "consultation_limit"
    elif not state.get("messages") or explicit_stop(latest_text(state)):
        plan = fallback_plan(state)
        plan.stop_requested = explicit_stop(latest_text(state))
        plan.closure_reason = "opt_out" if plan.stop_requested else None
        plan.next_advisor = "respond" if "exit" in consulted else "exit"
    elif "exit" in consulted and should_close(state):
        plan = Plan(**{**previous, "next_advisor": "respond"})
    else:
        plan = backend.plan(state)
    if state.get("selected_slot_id") and not count and not plan.stop_requested:
        plan.scheduling_needed = True
        plan.next_advisor = "scheduling"
    old_intent = state.get("turn_intent", {})
    intent = {
        "stop_requested": old_intent.get("stop_requested", False) or plan.stop_requested,
        "scheduling_needed": old_intent.get("scheduling_needed", False) or plan.scheduling_needed,
        "info_question": plan.info_question or old_intent.get("info_question"),
        "closure_reason": plan.closure_reason or old_intent.get("closure_reason"),
    }
    if reason is None:
        if intent["stop_requested"]:
            plan.next_advisor = "respond" if "exit" in consulted else "exit"
        elif plan.next_advisor == "respond":
            plan.next_advisor = pending_advisor(state, intent) or (
                "exit" if not count else "respond"
            )
        if plan.next_advisor != "respond" and task_fingerprint(
            state, plan.model_dump()
        ) in state.get("consulted_tasks", []):
            remaining = pending_advisor(state, intent)
            alternative = plan.model_copy(update={"next_advisor": remaining}) if remaining else None
            if alternative and task_fingerprint(state, alternative.model_dump()) not in state.get(
                "consulted_tasks", []
            ):
                rerouted_from = plan.next_advisor
                plan.next_advisor = remaining
            else:
                plan.next_advisor = "respond"
                reason = "repeated_advisor"
    offered_ids = {slot["id"] for slot in state.get("offered_slots", [])}
    declined = set(state.get("declined_slot_ids", [])) | (set(plan.declined_slot_ids) & offered_ids)
    return {
        "plan": plan.model_dump(),
        "declined_slot_ids": sorted(declined),
        "turn_intent": intent,
        "routing_stop_reason": reason,
        "routing_history": [
            *state.get("routing_history", []),
            {
                "next_advisor": plan.next_advisor,
                "after_consultations": count,
                "stop_reason": reason,
                "rerouted_from": rerouted_from,
            },
        ],
    }


def should_close(state: dict) -> bool:
    plan = state.get("turn_intent") or state.get("plan", {})
    if plan.get("stop_requested"):
        # A profile fact or a Main Agent assertion alone cannot authorize closure.
        return explicit_stop(latest_text(state)) or (
            plan.get("closure_reason") == "opt_out" and state.get("exit_label") == "end"
        )
    # Exit's binary label also covers completed scheduling. Without an opt-out
    # intent it cannot establish closure; only a SQL receipt establishes completion.
    return explicit_stop(latest_text(state))


def compose(state: dict) -> dict:
    """The Main Agent owns the action; factual booking text comes only from tool receipts."""
    plan = state["plan"]
    intent = state.get("turn_intent") or plan
    schedule = state.get("schedule", {})
    info = state.get("info", {})
    booking = schedule.get("booking") or state.get("booking")
    slots = schedule.get("slots", state.get("offered_slots", []))
    slots = [slot for slot in slots if slot["id"] not in state.get("declined_slot_ids", [])]
    scheduling_performed = schedule.get("decision") == "schedule"
    sources = info.get("source_ids", [])
    action, status = "continue", "active"
    closure_reason = None
    if should_close(state):
        action, status = "end", "closed"
        closure_reason = "opt_out"
        response = "Thanks for letting me know. I'll end this conversation here. Best wishes!"
        slots = []
    elif state.get("error"):
        response = "I couldn't complete that request right now. Please try again."
        if booking:
            response = f"Your interview is confirmed for {booking['slot']['label']}. I couldn't answer the remaining question just now; please try again."
    else:
        parts = []
        if booking and (
            schedule.get("status") in ("booked", "already_booked")
            or state.get("exit_label") == "end"
        ):
            parts.append(f"Your interview is confirmed for {booking['slot']['label']}.")
            if not intent.get("info_question"):
                action, status = "end", "closed"
                closure_reason = "scheduled"
                parts.append("Looking forward to speaking with you!")
        elif scheduling_performed:
            action = "schedule" if state.get("scheduling_enabled", True) else "continue"
            if schedule.get("message"):
                parts.append(schedule["message"])
            if slots:
                parts.append(
                    "Available appointments:\n"
                    + "\n".join(f"{index}. {slot['label']}" for index, slot in enumerate(slots, 1))
                )
                parts.append("Which appointment would you like to confirm?")
        if info:
            parts.append(info["answer"])
            action = "schedule" if slots and scheduling_performed else "continue"
            if not scheduling_performed and not booking and state.get("scheduling_enabled", True):
                parts.append("Would you like to arrange an interview?")
        if not parts:
            conflicts = state.get("candidate_profile", {}).get("conflicts", [])
            fields = {item["field"] for item in conflicts}
            question = (
                "Your background contains different Python experience claims. Please state your current years of Python experience and explicitly correct the older claim."
                if "python_experience" in fields
                else "Your sources contain different names. Please state your correct name explicitly."
                if "identity" in fields
                else "Would you like to end the conversation, or continue arranging an interview?"
                if state.get("exit_label") == "end"
                else plan["next_question"]
            )
            parts.append(question)
        if state.get("routing_stop_reason") in {
            "consultation_limit",
            "repeated_advisor",
        } and pending_advisor(state, intent):
            parts.append(
                "I still need to check the remaining request. Please tell me which question or interview time you'd like to handle next."
            )
        response = "\n\n".join(parts)
    result = TurnResult(
        action=action,
        response=response,
        conversation_status=status,
        booking=booking,
        offered_slots=slots,
        source_ids=sources,
        advisors=state.get("advisors", []),
        error=state.get("error"),
        closure_reason=closure_reason,
    )
    messages = [*state.get("messages", []), {"role": "assistant", "content": response}]
    return {
        "result": result.model_dump(),
        "messages": messages,
        "booking": booking,
        "offered_slots": slots,
        "conversation_status": status,
    }
