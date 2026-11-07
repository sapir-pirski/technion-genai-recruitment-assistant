import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

from openai import APIStatusError, OpenAI

from app.agents.backends.common import instruction_role, prompt
from app.config import ROOT, Settings
from app.observability.logging import log_event


def _cleanup_uploads(client, file_ids):
    remaining = []
    for file_id in file_ids:
        try:
            client.files.delete(file_id)
        except Exception as error:
            if isinstance(error, APIStatusError) and error.status_code == 404:
                continue
            remaining.append(file_id)
            log_event(
                "fine_tuning_cleanup_failed",
                level=logging.WARNING,
                component="app.fine_tuning",
                error_code=type(error).__name__,
            )
    return remaining


def _record_failure(stage, error, uploaded, remaining, *, outcome_unknown):
    # File IDs permit recovery; no source contents or provider error messages are saved.
    try:
        path = ROOT / "artifacts/fine-tuning-submission.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "stage": stage,
                    "error_code": type(error).__name__,
                    "uploaded_file_ids": uploaded,
                    "remaining_file_ids": remaining,
                    "job_outcome_unknown": outcome_unknown,
                    "upload_outcome_unknown": stage in {"training_upload", "validation_upload"},
                },
                handle,
                indent=2,
            )
        path.chmod(0o600)
    except Exception as diagnostic_error:
        log_event(
            "fine_tuning_diagnostic_failed",
            level=logging.WARNING,
            component="app.fine_tuning",
            error_code=type(diagnostic_error).__name__,
        )


def _metadata(job, base_model, files):
    return {
        "job_id": job.id,
        "status": job.status,
        "base_model": base_model,
        "training_file": files[0],
        "validation_file": files[1],
        "epochs": 3,
    }


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
    uploaded = []
    stage = "training_upload"
    try:
        with training.open("rb") as handle:
            uploaded.append(client.files.create(file=handle, purpose="fine-tune").id)
        stage = "validation_upload"
        with validation.open("rb") as handle:
            uploaded.append(client.files.create(file=handle, purpose="fine-tune").id)
        stage = "job_creation"
        job = client.fine_tuning.jobs.create(
            training_file=uploaded[0],
            validation_file=uploaded[1],
            model=base_model,
            suffix="recruitment-exit",
            method={"type": "supervised", "supervised": {"hyperparameters": {"n_epochs": 3}}},
        )
    except Exception as error:
        definitive_rejection = isinstance(error, APIStatusError) and error.status_code in {
            400,
            401,
            403,
            404,
            405,
            413,
            415,
            422,
        }
        unknown = stage == "job_creation" and not definitive_rejection
        if unknown:
            # A lost creation response can still correspond to a running paid job.
            # Recover a matching job when visible; absence from this page is not proof
            # of rejection, so preserve its files rather than risk breaking training.
            try:
                for candidate in client.fine_tuning.jobs.list(limit=100).data:
                    if (
                        candidate.training_file == uploaded[0]
                        and candidate.validation_file == uploaded[1]
                        and candidate.model == base_model
                    ):
                        return {**_metadata(candidate, base_model, uploaded), "recovered": True}
            except Exception as lookup_error:
                log_event(
                    "fine_tuning_recovery_failed",
                    level=logging.WARNING,
                    component="app.fine_tuning",
                    error_code=type(lookup_error).__name__,
                )
        remaining = list(uploaded) if unknown else _cleanup_uploads(client, uploaded)
        _record_failure(stage, error, uploaded, remaining, outcome_unknown=unknown)
        raise
    return _metadata(job, base_model, uploaded)


def status(settings: Settings, job_id: str) -> dict:
    client = OpenAI(api_key=settings.require_api_key(), timeout=settings.request_timeout)
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "job_id": job.id,
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "trained_tokens": job.trained_tokens,
    }
