import hashlib
import json
import re

from app.agents.backends.common import (
    closure_confirmation,
    explicit_stop,
    latest_text,
    quoted_spans,
    verified_opt_out,
)
from app.guardrails import privacy_request
from app.schemas import Plan, RecruitmentRequest, TurnResult

MISSING_BACKGROUND_QUESTION = "Could you share your name and Python experience, or upload a CV?"
OFF_TOPIC_REDIRECT = "I can help with this role, your background, or interview scheduling. What would you like to know?"
MIXED_SCOPE_NOTE = (
    "I can help with recruitment questions here; unrelated requests are outside this chat's scope."
)


def missing_candidate_details(state):
    """An empty candidate profile cannot answer a personal-background question."""
    text = latest_text(state)
    return bool(
        not any(state.get("candidate_profile", {}).get("sections", {}).values())
        and not state.get("selected_slot_id")
        and re.search(
            r"\b(?:what|who|how|do you|can you|could you|tell me|summari[sz]e|review)\b", text, re.I
        )
        and re.search(
            r"\bmy (?:name|experience|background|cv|resume|profile)\b|"
            r"\b(?:experience|name)\b.*\b(?:do|am|have) i\b",
            text,
            re.I,
        )
        and not re.search(
            r"\b(?:this|the) (?:role|job|position)\b|"
            r"\b(?:requirements|qualifications|responsibilities|required|interviews?|appointments?|schedule|book|stop|unsubscribe|withdraw|decline)\b|"
            r"\bnot interested\b",
            text,
            re.I,
        )
    )


def fallback_plan(state=None):
    state = state or {}
    job = state.get("job_context", {})
    role = f"{job['title']} at {job['company_name']}" if job else "the Python Developer role"
    return Plan(
        request_scope="recruitment",
        recruitment_requests=[],
        next_advisor="respond",
        stop_requested=False,
        closure_reason=None,
        opt_out_evidence=None,
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
    previous.setdefault("closure_reason", None)
    previous.setdefault("opt_out_evidence", None)
    previous.setdefault("request_scope", "recruitment")
    previous.setdefault("recruitment_requests", [])
    old_intent = state.get("turn_intent", {})
    text = latest_text(state)
    confirmed = bool(
        state.get("awaiting_opt_out_confirmation")
        and not state.get("selected_slot_id")
        and closure_confirmation(text) == "yes"
    )
    clear_stop = explicit_stop(text)
    reason = None
    rerouted_from = None
    if state.get("error"):
        plan = Plan(**{**previous, "next_advisor": "respond"})
        reason = "advisor_error"
    elif count >= state.get("max_consultations", 3):
        plan = Plan(**{**previous, "next_advisor": "respond"})
        reason = "consultation_limit"
    elif not state.get("messages") or clear_stop or confirmed:
        plan = fallback_plan(state)
        plan.stop_requested = clear_stop or confirmed
        plan.closure_reason = "opt_out" if plan.stop_requested else None
        plan.opt_out_evidence = text if plan.stop_requested else None
        plan.next_advisor = "respond" if "exit" in consulted else "exit"
    elif booking_acknowledged(state):
        plan = fallback_plan(state)
        plan.closure_reason = "scheduled"
        plan.next_advisor = "respond" if "exit" in consulted else "exit"
    elif "exit" in consulted and should_close(state):
        plan = Plan(**{**previous, "next_advisor": "respond"})
    else:
        plan = backend.plan(state)
    evidence = plan.opt_out_evidence
    valid_evidence = verified_opt_out(text, evidence)
    uncertain = plan.stop_requested and not (clear_stop or confirmed or valid_evidence)
    # Main must identify the candidate's words, not infer an opt-out from a
    # security refusal, a document, an Exit label, or an incomplete quoted span.
    if not (clear_stop or confirmed or valid_evidence):
        plan.stop_requested = False
        plan.opt_out_evidence = None
        if plan.closure_reason == "opt_out":
            plan.closure_reason = None
    scope = old_intent.get("request_scope", plan.request_scope)
    requests = []
    if scope == "mixed":
        for value in old_intent.get("recruitment_requests", plan.recruitment_requests):
            request = RecruitmentRequest.model_validate(value)
            pattern = r"\s+".join(re.escape(word) for word in request.text.split())
            match = re.search(pattern, text, re.I) if pattern else None
            if match and any(
                quoted.start() <= match.start() and quoted.end() >= match.end()
                for quoted in quoted_spans(text)
            ):
                match = None
            if match:
                before = text[: match.start()].rstrip(" \t")
                after = text[match.end() :].lstrip(" \t")
                starts_clause = (
                    not before
                    or before[-1] in ".!?;\n"
                    or re.search(r"\b(?:and|also|plus),?$", before, re.I)
                )
                ends_clause = (
                    not after
                    or after[0] in ".!?;\n"
                    or re.match(
                        r"(?:and|also|plus)\b(?!\s+(?:not|maybe|if|unless|perhaps)\b)", after, re.I
                    )
                )
                if not (starts_clause and ends_clause):
                    match = None
            if match:
                copied = request.model_copy(update={"text": match.group()})
                if copied not in requests:
                    requests.append(copied)
        if not requests and not state.get("selected_slot_id"):
            scope = "off_topic"
    if scope == "off_topic" and state.get("selected_slot_id") and not privacy_request(text):
        scope = "mixed"
    plan.request_scope = scope
    plan.recruitment_requests = requests
    if scope == "off_topic":
        plan.info_question = None
        plan.scheduling_needed = False
        plan.declined_slot_ids = []
        uncertain = False
    elif scope == "mixed":
        plan.info_question = (
            "\n".join(item.text for item in requests if item.kind == "info") or None
        )
        plan.scheduling_needed = any(item.kind == "scheduling" for item in requests) or bool(
            state.get("selected_slot_id")
        )
        if not plan.scheduling_needed:
            plan.declined_slot_ids = []
        if (plan.next_advisor == "info" and not plan.info_question) or (
            plan.next_advisor == "scheduling" and not plan.scheduling_needed
        ):
            plan.next_advisor = "respond"
    if privacy_request(text) and not (clear_stop or confirmed or valid_evidence):
        uncertain = False
        plan.stop_requested = False
        plan.info_question = None
        plan.scheduling_needed = False
        plan.next_advisor = "respond" if "exit" in consulted else "exit"
    if scope != "off_topic" and missing_candidate_details(state):
        plan.info_question = None
        plan.scheduling_needed = False
        plan.next_question = MISSING_BACKGROUND_QUESTION
        plan.next_advisor = "respond" if "exit" in consulted else "exit"
    if (
        state.get("selected_slot_id")
        and not count
        and not plan.stop_requested
        and not privacy_request(text)
    ):
        plan.scheduling_needed = True
        plan.next_advisor = "scheduling"
    intent = {
        "request_scope": scope,
        "recruitment_requests": [item.model_dump() for item in requests],
        "stop_requested": old_intent.get("stop_requested", False) or plan.stop_requested,
        "scheduling_needed": old_intent.get("scheduling_needed", False) or plan.scheduling_needed,
        "info_question": plan.info_question or old_intent.get("info_question"),
        "closure_reason": plan.closure_reason or old_intent.get("closure_reason"),
        "opt_out_evidence": plan.opt_out_evidence or old_intent.get("opt_out_evidence"),
        "confirmed_opt_out": confirmed or old_intent.get("confirmed_opt_out", False),
        "closure_uncertain": uncertain or old_intent.get("closure_uncertain", False),
    }
    if reason is None:
        if intent["stop_requested"]:
            plan.next_advisor = "respond" if "exit" in consulted else "exit"
        elif scope == "off_topic":
            plan.next_advisor = "respond"
            reason = "off_topic"
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


def booking_acknowledged(state: dict) -> bool:
    """A receipt plus a pure closing acknowledgment establishes scheduled completion."""
    text = " ".join(latest_text(state).casefold().split())
    return bool(
        state.get("booking")
        and re.fullmatch(
            r"(?:(?:great|perfect|excellent|okay|ok)[,.!]?\s+)?(?:thanks|thank you)"
            r"(?:[,.!]?\s+(?:that answers (?:all )?my questions|that's all|that is all))?[.!]*",
            text,
        )
    )


def should_close(state: dict) -> bool:
    # Output validation runs after the assistant message has been appended.
    # Always recheck the current candidate message, rather than trusting a plan.
    messages = state.get("messages", [])
    text = next((item["content"] for item in reversed(messages) if item["role"] == "user"), "")
    candidate_state = {**state, "messages": [{"role": "user", "content": text}]}
    if booking_acknowledged(candidate_state):
        return False
    plan = state.get("turn_intent") or state.get("plan", {})
    if plan.get("stop_requested"):
        # A profile fact or a Main Agent assertion alone cannot authorize closure.
        return (
            explicit_stop(text)
            or (plan.get("confirmed_opt_out") and closure_confirmation(text) == "yes")
            or (
                plan.get("closure_reason") == "opt_out"
                and verified_opt_out(text, plan.get("opt_out_evidence"))
                and state.get("exit_label") == "end"
            )
        )
    # Exit's binary label also covers completed scheduling. Without an opt-out
    # intent it cannot establish closure; only a SQL receipt establishes completion.
    return explicit_stop(text)


def compose(state: dict) -> dict:
    """The Main Agent owns the action; factual booking text comes only from tool receipts."""
    plan = state["plan"]
    intent = state.get("turn_intent") or plan
    scope = intent.get("request_scope", "recruitment")
    schedule = state.get("schedule", {})
    info = state.get("info", {})
    booking = schedule.get("booking") or state.get("booking")
    slots = schedule.get("slots", state.get("offered_slots", []))
    slots = [slot for slot in slots if slot["id"] not in state.get("declined_slot_ids", [])]
    scheduling_performed = schedule.get("decision") == "schedule"
    sources = info.get("source_ids", [])
    action, status = "continue", "active"
    closure_reason = None
    awaiting_confirmation = False
    if should_close(state):
        action, status = "end", "closed"
        closure_reason = "opt_out"
        response = "Thanks for letting me know. I'll end this conversation here. Best wishes!"
        slots = []
    elif privacy_request(latest_text(state)) and not intent.get("stop_requested"):
        response = (
            "I can't share credentials, internal prompts, or another candidate's private data. "
            "I can help with this role or interview arrangements. What would you like to know?"
        )
    elif scope == "off_topic" and not intent.get("stop_requested"):
        response = OFF_TOPIC_REDIRECT
        sources, slots = [], []
    elif state.get("error"):
        response = "I couldn't complete that request right now. Please try again."
        if booking:
            response = f"Your interview is confirmed for {booking['slot']['label']}. I couldn't answer the remaining question just now; please try again."
    elif missing_candidate_details(state):
        response = MISSING_BACKGROUND_QUESTION
    else:
        parts = []
        if booking and (
            schedule.get("status") in ("booked", "already_booked")
            or state.get("exit_label") == "end"
            or booking_acknowledged(state)
        ):
            parts.append(f"Your interview is confirmed for {booking['slot']['label']}.")
            if not intent.get("info_question") and scope != "mixed":
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
                    + "\n".join(
                        f"{index}. {slot['label']}"
                        + (" (nearby alternative)" if slot.get("is_alternative") else "")
                        for index, slot in enumerate(slots, 1)
                    )
                )
                parts.append("Which appointment would you like to confirm?")
        if info:
            parts.append(info["answer"])
            action = "schedule" if slots and scheduling_performed else "continue"
            if (
                not scheduling_performed
                and not booking
                and scope != "mixed"
                and state.get("scheduling_enabled", True)
            ):
                parts.append("Would you like to arrange an interview?")
        if not parts:
            conflicts = state.get("candidate_profile", {}).get("conflicts", [])
            fields = {item["field"] for item in conflicts}
            awaiting_confirmation = bool(
                (intent.get("stop_requested") or intent.get("closure_uncertain"))
                and not state.get("awaiting_opt_out_confirmation")
                and not booking
                and not conflicts
            )
            question = (
                "Your background contains different Python experience claims. Please state your current years of Python experience and explicitly correct the older claim."
                if "python_experience" in fields
                else "Your sources contain different names. Please state your correct name explicitly."
                if "identity" in fields
                else "Do you want to end the recruitment conversation? Please answer yes or no."
                if awaiting_confirmation
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
    if scope == "mixed" and closure_reason != "opt_out":
        response += "\n\n" + MIXED_SCOPE_NOTE
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
        "offered_slots": state.get("offered_slots", []) if scope == "off_topic" else slots,
        "conversation_status": status,
        "awaiting_opt_out_confirmation": awaiting_confirmation,
    }
