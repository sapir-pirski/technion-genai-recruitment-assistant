import hashlib
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
_QUOTED_TEXT = re.compile(r'```[\s\S]*?```|`[^`]*`|"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)\'[^\']*\'(?!\w)')
OPERATION_PROMPTS = {
    "main": "main",
    "info": "info",
    "info_qualifications": "info_qualifications",
    "scheduling": "scheduling",
    "info_decision": "info_decision",
    "exit": "exit",
    "profile_extraction": "profile",
    "chat_profile": "chat_profile",
    "followups": "followups",
    "judge": "judge",
}


def prompt(name: str) -> str:
    return (PROMPTS / f"{name}.txt").read_text()


def instruction_role(model: str) -> str:
    return "developer" if model in {"gpt-5-nano", "gpt-5-nano-2025-08-07"} else "system"


def prompt_spec(name: str, model: str) -> dict:
    content = prompt(name)
    return {
        "prompt_name": name,
        "prompt_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "instruction_role": instruction_role(model),
        "content": content,
    }


def prompt_manifest(settings) -> dict:
    manifest = {
        operation: {
            key: value
            for key, value in prompt_spec(name, settings.model_for(operation)).items()
            if key != "content"
        }
        for operation, name in OPERATION_PROMPTS.items()
    }
    if settings.exit_provider == "huggingface":
        from app.fine_tuning.exit_data import classifier_spec

        manifest["exit"] = classifier_spec(settings.hf_exit_model, settings.hf_exit_revision)
    return manifest


def latest_text(state: dict) -> str:
    messages = state.get("messages", [])
    return messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""


def model_context(state: dict, *, include_profile=True) -> dict:
    # Deliberately excludes labels, evaluation targets, and internal credentials.
    profile = state.get("candidate_profile", {})
    context = {
        "history": state.get("messages", []),
        "job": state.get("job_context", {}),
        "candidate_profile": {
            key: value
            for key, value in profile.items()
            if key in {"sections", "conflicts", "missing_information", "verification"}
        },
        "profile_status": state.get("profile_status", "empty"),
        "profile_issues": [
            {"source_kind": issue["source_kind"], "error_code": issue["error_code"]}
            for issue in state.get("profile_issues", [])
        ],
        "reference_time": state["reference_time"],
        "timezone": state["timezone"],
        "offered_slots": state.get("offered_slots", []),
        "selected_slot_id": state.get("selected_slot_id"),
        "schedule_preferences": state.get("schedule_preferences", {}),
        "booking": state.get("booking"),
        "advisor_results": state.get("advisor_results", []),
        "turn_intent": state.get("turn_intent", {}),
        "awaiting_opt_out_confirmation": state.get("awaiting_opt_out_confirmation", False),
        "remaining_consultations": max(
            0, state.get("max_consultations", 3) - state.get("consultation_count", 0)
        ),
        "input_kind": state.get("input_kind", "chat"),
    }
    if not include_profile:
        for key in ("candidate_profile", "profile_status", "profile_issues"):
            context.pop(key, None)
    return context


def quoted_spans(text):
    return _QUOTED_TEXT.finditer(text)


def unquoted_text(text):
    return _QUOTED_TEXT.sub(" ", text)


def explicit_stop(text: str) -> bool:
    # Match complete, unambiguous requests. Contextual statements go to Exit Advisor.
    normalized = " ".join(text.lower().replace("’", "'").split())
    return bool(
        re.fullmatch(
            r"(?:(?:thanks|thank you)(?: for (?:the|your) (?:offer|opportunity|time))?[,!.]? |"
            r"i appreciate (?:the|your) (?:offer|opportunity|time)[,.]? )?"
            r"(?:but[, ]+)?(?:please )?(?:stop|unsubscribe|quit|end (?:this |the )?conversation|"
            r"stop (?:texting|messaging|contacting)(?: me)?(?: and (?:cancel|end) (?:this|the) conversation)?|"
            r"(?:don't|do not) contact me(?: again)?|"
            r"remove me(?: from (?:this|the|your) (?:recruitment process|contact list|list))?|"
            r"(?:i(?:'m| am) )?(?:not|no longer) interested(?: in (?:this|the) (?:role|position|job))?|"
            r"i (?:will not|won't|do not want to|don't want to) (?:pursue|continue with) "
            r"(?:this|the) (?:role|position|job|application|recruitment process)|"
            r"i(?:'d| would) like to withdraw (?:my application|from (?:this|the) (?:role|process))|"
            r"i(?:'ve| have)? decided (?:not to (?:pursue|continue with) (?:this|the) "
            r"(?:role|position|job|application)|to withdraw my application)|"
            r"i(?:'ll| will) pass on (?:this|the) (?:role|position|job|opportunity)|"
            r"i(?:'ve| have)? (?:accepted|found) (?:another|a) (?:job|offer))"
            r"(?:[.!]|,? (?:please|thanks|thank you))?[.!]?",
            normalized,
        )
    )


def scoped_advisor_state(state, advisor=None):
    """Keep mixed-request advisors on the candidate's relevant, verified clauses."""
    intent = state.get("turn_intent", {})
    if intent.get("request_scope") != "mixed":
        return state
    text = "\n".join(
        item["text"]
        for item in intent.get("recruitment_requests", [])
        if item["kind"] != "exit" and (advisor is None or item["kind"] == advisor)
    )
    if not text and advisor in {None, "scheduling"} and state.get("selected_slot_id"):
        text = "Please confirm the selected appointment."
    if not text:
        return state
    messages = list(state.get("messages", []))
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            original = messages[index]["content"]
            messages[index] = {"role": "user", "content": text}
            break
    else:
        return state
    return {
        **state,
        "unscoped_candidate_text": original,
        "messages": messages,
    }


def verified_opt_out(text: str, evidence: str | None) -> bool:
    """Accept only a verbatim, complete candidate clause expressing withdrawal."""
    if not evidence or evidence not in text:
        return False
    if re.search(
        r"\b(?:if|suppose|hypothetical|example|quoted?|says?|said|wrote|reads?)\b", text, re.I
    ):
        return False
    # A quoted instruction or truncated span cannot authorize an action.
    unquoted = re.sub(r"^>.*$", "", unquoted_text(text), flags=re.M)
    clauses = re.split(r"[.!?;\n]+", unquoted)
    return any(
        evidence.strip().rstrip(".!?;") == clause.strip() and explicit_stop(clause.strip())
        for clause in clauses
    )


def closure_confirmation(text: str) -> str | None:
    normalized = " ".join(text.casefold().replace("’", "'").split()).strip(".! ")
    if normalized in {"yes", "yes please", "yes, please", "yes, stop", "yes, end it"}:
        return "yes"
    if normalized in {"no", "no thanks", "no, thanks", "continue", "please continue"}:
        return "no"
    return None
