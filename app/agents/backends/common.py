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


def model_context(state: dict) -> dict:
    # Background is supplied in conversation history; labels and credentials stay out.
    return {
        "history": state.get("messages", []),
        "job": state.get("job_context", {}),
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
    }


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


def has_opt_out_signal(text: str) -> bool:
    """Ground a contextual opt-out in the candidate's words, excluding quoted text."""
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]*\x27(?!\w)', "", text)
    normalized = " ".join(text.casefold().replace("’", "'").split())
    return bool(
        re.search(
            r"\b(?:not interested|no longer interested|not (?:currently )?looking for|"
            r"(?:stop|don't|do not) (?:contacting|messaging|texting|contact|message|text) me|"
            r"(?:withdraw|cancel) my (?:application|candidacy)|"
            r"(?:end|leave) (?:this |the )?conversation|"
            r"(?:i have|i've|i) (?:accepted|found) (?:another|a) (?:job|offer)|"
            r"(?:won't|will not|don't want to) (?:pursue|proceed)|no (?:thanks|thank you))\b",
            normalized,
        )
    )
