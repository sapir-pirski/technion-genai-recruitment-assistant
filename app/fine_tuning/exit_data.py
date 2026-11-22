"""One input contract for local Exit training and classifier inference."""

import hashlib
import json

from app.evaluation.dataset import redact
from app.guardrails import model_data

LABELS = ("do_not_end", "end")
INPUT_FORMAT = "exit-history-json-v1: role/content JSON; last 12 messages; 16000 chars; redact phones/emails/credentials; left token truncation"
INPUT_SHA256 = hashlib.sha256(INPUT_FORMAT.encode()).hexdigest()


def history_text(history: list[dict]) -> str:
    if any(
        message.get("role") not in {"user", "assistant"}
        or not isinstance(message.get("content"), str)
        for message in history
    ):
        raise ValueError("Exit history must contain candidate and assistant text messages.")
    history = [
        {"role": message["role"], "content": redact(message["content"])} for message in history
    ]
    packet = model_data({"history": history}, operation="exit")
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


def classifier_spec(model: str, revision: str) -> dict:
    return {
        "prompt_name": "exit_classifier",
        "prompt_sha256": INPUT_SHA256,
        "instruction_role": None,
        "input_format": INPUT_FORMAT,
        "model": model,
        "model_revision": revision,
    }
