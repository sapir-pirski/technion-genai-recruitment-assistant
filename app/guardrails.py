"""Deterministic checks at input, model, tool and output boundaries.

These checks enforce concrete invariants. They do not certify semantic factuality
or detect every secret or prompt-injection attempt.
"""

import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from app.observability.logging import log_event
from app.observability.tracing import span
from app.schemas import TurnResult

MAX_MESSAGE_CHARS = 4000
MAX_HISTORY_CHARS = 16000
MAX_MODEL_CHARS = 48000
MAX_RESPONSE_CHARS = 16000

_CREDENTIALS = re.compile(
    r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?-----END (?:[A-Z]+ )?PRIVATE KEY-----"
    r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|\bhf_[A-Za-z0-9]{20,}\b"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
    r"|\bBearer\s+[A-Za-z0-9_.~-]{20,}"
    r"|\b(?:api[_ -]?key|password|client[_ -]?secret)\s*[:=]\s*[\"']?[^\s\"',;]{8,}"
    r"|\b[a-z][a-z0-9+.-]*://[^\s/:@]+:[^\s/@]+@",
    re.IGNORECASE,
)
_BOOKING_CLAIM = re.compile(
    r"\b(?:interview|appointment|booking)\s+(?:is\s+|has been\s+)?"
    r"(?:confirmed|booked|scheduled)\b|\b(?:I(?:'ve| have)?|we(?:'ve| have)?)\s+"
    r"(?:confirmed|booked|scheduled)\s+(?:your|the)\s+(?:interview|appointment)\b",
    re.IGNORECASE,
)
_UNSAFE_QUESTION = re.compile(
    r"^(?:please\s+)?(?:book|confirm|cancel|accept|stop|unsubscribe)\b"
    r"|\b(?:can|could|will|would)\s+(?:you|we)\s+(?:please\s+)?"
    r"(?:book|confirm|cancel|accept|stop)\b"
    r"|\b(?:password|api[ _-]?key|private key|credit card|social security|passport number)\b",
    re.IGNORECASE,
)


class GuardrailViolation(ValueError):
    """A static, user-safe failure without the rejected content."""


def record(rule_id, stage, outcome, reason_code):
    attributes = dict(rule_id=rule_id, stage=stage, outcome=outcome, reason_code=reason_code)
    with span("guardrail", stage, attributes=attributes):
        log_event("guardrail_checked", component="app.guardrails", **attributes)


def reject(rule_id, stage, reason_code, message):
    record(rule_id, stage, "block", reason_code)
    raise GuardrailViolation(message)


def _redact(value):
    if isinstance(value, str):
        return _CREDENTIALS.sub("[REDACTED_CREDENTIAL]", value)
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    return value


def redact_credentials(value, *, stage):
    cleaned = _redact(value)
    if cleaned != value:
        record("credentials", stage, "redact", "recognized_credential")
    return cleaned


def input_text(value):
    if value is not None and (not value.strip() or len(value) > MAX_MESSAGE_CHARS):
        reject("message_size", "input", "invalid_length", "Use a message of 1–4,000 characters.")
    return redact_credentials(value, stage="input")


def input_state(state):
    messages = state.get("messages", [])
    if any(
        message.get("role") not in {"user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in messages
    ):
        reject("dialogue_roles", "input", "invalid_role", "Use candidate and assistant messages.")
    if messages and messages[-1]["role"] == "user":
        input_text(messages[-1]["content"])
    record("dialogue_roles", "input", "allow", "valid_dialogue")
    return {"messages": redact_credentials(messages, stage="input")}


def model_data(value, *, operation):
    """Bound transmitted context; keep full transcript in application storage."""
    cleaned = redact_credentials(value, stage="model_input")
    # Quality packets must retain their exact evidence; reject rather than alter them.
    if operation == "judge" and cleaned != value:
        reject(
            "judge_evidence",
            "model_input",
            "credential_in_packet",
            "Remove credentials from the quality packet before evaluation.",
        )
    if isinstance(cleaned, dict) and "history" in cleaned and operation != "judge":
        history, size = [], 0
        for message in reversed(cleaned["history"][-12:]):
            length = len(json.dumps(message, ensure_ascii=False))
            if size + length > MAX_HISTORY_CHARS:
                break
            history.append(message)
            size += length
        history.reverse()
        if len(history) != len(cleaned["history"]):
            record("context_budget", "model_input", "redact", "older_history_omitted")
        cleaned = {**cleaned, "history": history}
    maximum = 50000 if operation == "judge" else MAX_MODEL_CHARS
    size = (
        len(cleaned) if isinstance(cleaned, str) else len(json.dumps(cleaned, ensure_ascii=False))
    )
    if size > maximum:
        reject(
            "context_budget",
            "model_input",
            "context_too_large",
            "The supplied context is too large. Use shorter background information.",
        )
    record("context_budget", "model_input", "allow", "within_budget")
    return cleaned


def candidate_accepts(text):
    """Conservative English consent grammar; ambiguous wording asks for confirmation."""
    text = text.replace("’", "'")
    # Quoted documents/code cannot supply acceptance. Preserve the candidate's own clauses.
    text = re.sub(r"```[\s\S]*?```|`[^`]*`|\"[^\"]*\"", " ", text)
    punctuation = re.search(r"[.!?;\n]", text)
    if punctuation and punctuation[0] == "?":
        return False
    if re.search(
        r"\b(?:don't|do not|never|cannot|can't)\s+(?:book|confirm|schedule|attend)\b"
        r"|\bnot available\b|\b(?:cancel|decline) (?:the|my|this|that) (?:appointment|interview)\b",
        text,
        re.I,
    ):
        return False
    first = re.split(r"[.!?;\n]", text.strip(), maxsplit=1)[0].strip().casefold()
    if re.search(r"\b(?:not|never|don't|do not|can't|cannot|cancel|decline|maybe|if|but)\b", first):
        return False
    if re.search(r"\b(?:job|salary)\b", first):
        return False
    return bool(
        re.fullmatch(r"(?:yes|yes please|okay|ok|that works|sounds good)(?:,?\s+please)?", first)
        or re.match(
            r"^(?:(?:yes|okay|ok)[, ]+)?(?:i (?:accept|agree)|please (?:book|confirm)|(?:book|confirm) (?:my|the|this|that)|that (?:time|appointment|slot) works)\b",
            first,
        )
        or re.fullmatch(
            r"(?:yes[, ]+)?(?:the )?(?:first|second|third)(?: (?:slot|appointment|option|one))?(?: works(?: for me)?)?",
            first,
        )
    )


def booking_allowed(state, selected, *, day=None, at=None, request=None):
    from app.agents.backends.common import explicit_stop, latest_text

    text = latest_text(state)
    offered = state.get("offered_slots", [])
    allowed = False
    if not explicit_stop(text):
        if state.get("selected_slot_id") == selected:
            allowed = True
        elif candidate_accepts(text):
            # A model-supplied slot ID must be independently resolved from candidate wording.
            ordinal = re.search(r"\b(first|second|third)\b", text, re.I)
            if ordinal:
                index = {"first": 0, "second": 1, "third": 2}[ordinal[1].casefold()]
                allowed = index < len(offered) and offered[index]["id"] == selected
            elif day is not None or at is not None:
                phrases = [v for v in (request.date_text, request.time_text) if v]
                matches = []
                for item in offered:
                    local = datetime.fromisoformat(item["starts_at"]).astimezone(
                        ZoneInfo(state["timezone"])
                    )
                    if (day is None or local.date() == day) and (
                        at is None or local.time().replace(tzinfo=None) == at
                    ):
                        matches.append(item["id"])
                allowed = (
                    bool(phrases)
                    and all(v.casefold() in text.casefold() for v in phrases)
                    and matches == [selected]
                )
            else:
                allowed = len(offered) == 1 and offered[0]["id"] == selected
    record(
        "booking_consent",
        "tool",
        "allow" if allowed else "clarify",
        "candidate_acceptance" if allowed else "acceptance_not_verified",
    )
    return allowed


def check_sources(source_ids, allowed):
    if not source_ids or not set(source_ids).issubset(allowed):
        reject(
            "source_references",
            "retrieval",
            "unknown_source",
            "The answer could not be linked to the retrieved job description.",
        )
    record("source_references", "retrieval", "allow", "retrieved_sources_only")


def check_followups(questions):
    if any(_UNSAFE_QUESTION.search(text) or _CREDENTIALS.search(text) for text in questions):
        reject(
            "suggestion_actions",
            "output",
            "unsafe_suggestion",
            "Use questions about the role or interview preparation.",
        )
    record("suggestion_actions", "output", "allow", "safe_questions")


def output_result(state, repository):
    """Validate before a composed result reaches a checkpoint, stream or transcript."""
    from app.agents.main_agent import should_close

    result = TurnResult.model_validate(state["result"])
    receipt = repository.get_booking(
        state["conversation_id"], state["timezone"], state.get("job_id")
    )
    reason = None
    if result.booking != receipt:
        reason = "booking_receipt_mismatch"
    elif not set(result.source_ids).issubset(state.get("retrieved_source_ids", [])):
        reason = "unknown_source"
    elif len(result.response) > MAX_RESPONSE_CHARS or not result.response.strip():
        reason = "invalid_response_length"
    elif (result.action == "end") != (result.conversation_status == "closed"):
        reason = "inconsistent_action"
    elif result.action == "end" and not (
        (result.closure_reason == "opt_out" and should_close(state))
        or (result.closure_reason == "scheduled" and receipt)
    ):
        reason = "unauthorized_closure"
    elif result.action != "end" and result.closure_reason is not None:
        reason = "inconsistent_closure_reason"
    elif result.action == "schedule" and (
        not state.get("scheduling_enabled", True)
        or state.get("schedule", {}).get("decision") != "schedule"
    ):
        reason = "unauthorized_schedule_action"
    # Receipt text is the only permitted confirmation claim, even if a booking exists.
    canonical = f"Your interview is confirmed for {receipt['slot']['label']}." if receipt else ""
    remainder = result.response.replace(canonical, "", 1) if canonical else result.response
    if _BOOKING_CLAIM.search(remainder):
        reason = "unverified_booking_claim"
    cleaned = redact_credentials(result.response, stage="output")
    if reason:
        record("response_consistency", "output", "clarify", reason)
        clarification = (
            "Please clarify your question or choose an offered interview time."
            if state.get("scheduling_enabled", True)
            else "Please clarify your question or contact the employer about the next steps."
        )
        result = result.model_copy(
            update={
                "action": "continue",
                "conversation_status": "active",
                "closure_reason": None,
                "booking": receipt,
                "response": (canonical + "\n\n" if canonical else "")
                + "I couldn't verify the remaining details. "
                + clarification,
                "source_ids": [],
                "offered_slots": [],
                "suggested_questions": [],
                "suggestion_source": "none",
            }
        )
    else:
        result = result.model_copy(update={"response": cleaned})
        record("response_consistency", "output", "allow", "consistent_receipts_and_sources")
    messages = list(state.get("messages", []))
    if messages and messages[-1]["role"] == "assistant":
        messages[-1] = {"role": "assistant", "content": result.response}
    return {
        "result": result.model_dump(),
        "messages": messages,
        "booking": result.booking,
        "offered_slots": result.offered_slots,
        "conversation_status": result.conversation_status,
        "answer_validated": True,
    }
