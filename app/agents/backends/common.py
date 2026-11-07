import hashlib
import re
from pathlib import Path

PROMPTS = Path(__file__).resolve().parents[2] / "prompts"
OPERATION_PROMPTS = {
    "main": "main",
    "info": "info",
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
    base = model.split(":")[1] if model.startswith("ft:") else model
    return "developer" if base in {"gpt-5-nano", "gpt-5-nano-2025-08-07"} else "system"


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
        "schedule_preferences": state.get("schedule_preferences", {}),
        "booking": state.get("booking"),
        "advisor_results": state.get("advisor_results", []),
        "turn_intent": state.get("turn_intent", {}),
        "remaining_consultations": max(
            0, state.get("max_consultations", 3) - state.get("consultation_count", 0)
        ),
        "input_kind": state.get("input_kind", "chat"),
    }
    if not include_profile:
        for key in ("candidate_profile", "profile_status", "profile_issues"):
            context.pop(key, None)
    return context


def explicit_stop(text: str) -> bool:
    # Match complete, unambiguous requests. Contextual statements go to Exit Advisor.
    normalized = " ".join(text.lower().replace("’", "'").split())
    return bool(
        re.fullmatch(
            r"(?:please )?(?:stop|unsubscribe|quit|end (?:this |the )?conversation|"
            r"stop (?:texting|messaging|contacting)(?: me)?(?: and (?:cancel|end) (?:this|the) conversation)?|"
            r"(?:don't|do not) contact me(?: again)?|"
            r"remove me(?: from (?:this|the|your) (?:recruitment process|contact list|list))?|"
            r"(?:i(?:'m| am) )?(?:not|no longer) interested(?: in (?:this|the) (?:role|position|job))?|"
            r"i(?:'ve| have)? (?:accepted|found) (?:another|a) (?:job|offer))"
            r"(?:[.!]|,? (?:please|thanks|thank you))?[.!]?",
            normalized,
        )
    )
