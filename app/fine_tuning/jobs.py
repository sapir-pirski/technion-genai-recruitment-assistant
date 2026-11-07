import json
from pathlib import Path

from openai import APIStatusError, OpenAI

from app.agents.backends.common import instruction_role, prompt
from app.config import Settings


def validate_jsonl(path: Path, minimum=1, *, base_model: str | None = None) -> int:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) < minimum:
        raise ValueError(f"Training requires at least {minimum} examples.")
    expected_role = instruction_role(base_model) if base_model else None
    instructions = prompt("exit")
    for row in rows:
        messages = row.get("messages", [])
        roles = [m.get("role") for m in messages]
        if (
            len(roles) != 3
            or roles[0] not in {"system", "developer"}
            or roles[1:] != ["user", "assistant"]
        ):
            raise ValueError("Expected instructions, history input, and target output messages.")
        if expected_role and roles[0] != expected_role:
            raise ValueError(
                "Training instruction role differs from the base model; regenerate prepare-data with --base-model."
            )
        if messages[0].get("content") != instructions:
            raise ValueError(
                "Training instructions differ from the runtime Exit prompt; regenerate prepare-data."
            )
        if messages[-1]["content"] not in ("end", "do_not_end"):
            raise ValueError("Invalid Exit Advisor label.")
        payload = json.loads(messages[1]["content"])
        if set(payload) != {"history"} or not isinstance(payload["history"], list):
            raise ValueError("Training input must contain only conversation history.")
    return len(rows)


def submit(settings: Settings, training: Path, validation: Path, base_model: str) -> dict:
    validate_jsonl(training, minimum=10, base_model=base_model)
    validate_jsonl(validation, base_model=base_model)
    # Identical prefixes (notably empty openings) carry no candidate-specific leakage.
    # Conversation isolation is enforced by the prepare-data manifest.
    client = OpenAI(
        api_key=settings.require_api_key(),
        timeout=settings.request_timeout,
        max_retries=settings.max_retries,
    )
    with training.open("rb") as handle:
        train_file = client.files.create(file=handle, purpose="fine-tune")
    with validation.open("rb") as handle:
        validation_file = client.files.create(file=handle, purpose="fine-tune")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=train_file.id,
            validation_file=validation_file.id,
            model=base_model,
            suffix="recruitment-exit",
            method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 3}}},
        )
    except APIStatusError as error:
        # A definitive rejection must not leave private orphaned training uploads.
        if 400 <= error.status_code < 500:
            for file_id in (train_file.id, validation_file.id):
                client.files.delete(file_id)
        raise
    return {
        "job_id": job.id,
        "status": job.status,
        "base_model": base_model,
        "training_file": train_file.id,
        "validation_file": validation_file.id,
        "epochs": 3,
    }


def status(settings: Settings, job_id: str) -> dict:
    client = OpenAI(api_key=settings.require_api_key(), timeout=settings.request_timeout)
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
    }
