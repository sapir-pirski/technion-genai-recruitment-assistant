import json
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy.schema import CreateSchema, DropSchema

from app.agents.backends.common import prompt_manifest
from app.config import Settings
from app.evaluation.dataset import (
    build_examples,
    dataset_fingerprint,
    load_conversations,
    split_examples,
)
from app.evaluation.metrics import summarize
from app.service import ChatService
from app.storage.database import make_engine
from app.storage.schedule_repository import ScheduleRepository


def evaluate(
    settings: Settings, *, split="test", task="actions", output: Path | None = None
) -> dict:
    settings.require_api_key()
    prompt_versions = prompt_manifest(settings)
    conversations = load_conversations(settings)
    examples = split_examples(build_examples(conversations), split)
    rows = []
    # Benchmark cannot mutate the interactive application's bookings/checkpoints.
    engine = make_engine(settings.database_url)
    schema = "evaluation_" + uuid4().hex
    with engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    try:
        repository = ScheduleRepository(
            engine.execution_options(schema_translate_map={None: schema})
        )
        service = ChatService(settings, repository=repository, checkpointer=InMemorySaver())
        if isinstance(getattr(service.backend, "prompt_manifest", None), dict):
            prompt_versions = service.backend.prompt_manifest
        try:
            for example in examples:
                started = time.perf_counter()
                error = None
                with service.traces.trace(
                    None,
                    str(uuid4()),
                    component="evaluation",
                    operation=task,
                    attributes={
                        "split": split,
                        "example_conversation_id": example["conversation_id"],
                        "example_turn_id": example["turn_id"],
                    },
                ) as trace:
                    if task == "exit":
                        expected = "end" if example["label"] == "end" else "do_not_end"
                        try:
                            prediction = service.backend.exit_label(example["history"])
                            if prediction not in {"end", "do_not_end"}:
                                raise ValueError("Invalid Exit label")
                        except Exception as exc:
                            prediction, error = None, type(exc).__name__
                    else:
                        expected = example["label"]
                        try:
                            result = service.evaluate_prefix(
                                example["history"],
                                datetime.fromisoformat(example["reference_time"]),
                            )
                            error = result.error
                            prediction = None if error else result.action
                        except Exception as exc:
                            prediction, error = None, type(exc).__name__
                    if error:
                        trace.fail(error)
                rows.append(
                    {
                        "conversation_id": example["conversation_id"],
                        "turn_id": example["turn_id"],
                        "expected": expected,
                        "predicted": prediction,
                        "error": error,
                        "is_opening": example["is_opening"],
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
        finally:
            service.close()
    finally:
        with engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        engine.dispose()
    labels = ["do_not_end", "end"] if task == "exit" else None
    report = {
        "provider": settings.exit_provider if task == "exit" else "openai",
        "model": settings.chat_model,
        "helper_model": settings.helper_model,
        "exit_model": settings.model_for("exit"),
        "exit_provider": settings.exit_provider,
        "exit_revision": settings.hf_exit_revision
        if settings.exit_provider == "huggingface"
        else None,
        "model_policy": settings.model_policy(),
        "prompt_manifest": prompt_versions,
        "task": task,
        "evaluation_mode": "historical_prefix",
        "booking_correctness": None,
        "calendar_context": "unavailable; empty isolated calendar",
        "metric_version": "failure-aware-v2",
        "split": split,
        "dataset_version": settings.dataset_version,
        "dataset_sha256": dataset_fingerprint(conversations),
        "created_at": datetime.now().astimezone().isoformat(),
        "all_turns": summarize(rows, labels),
        "candidate_responses": summarize([row for row in rows if not row["is_opening"]], labels),
        "rows": rows,
        "limitations": [
            "15 conversations; results are exploratory.",
            "Historical transcripts lack verified slot IDs. The runtime refuses unverified bookings; gold labels sometimes assume them.",
            "No verified historical recruiter calendar was supplied; evaluation uses an empty isolated calendar and never invents availability.",
        ],
    }
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        output.chmod(0o600)
    return report


def evaluate_local_exit(settings, artifact: Path, *, split="test", output: Path | None = None):
    """Evaluate a completed local HF artifact without uploads or OpenAI calls."""
    import hashlib

    from app.agents.backends.huggingface import exit_label

    conversations = load_conversations(settings)
    examples = split_examples(build_examples(conversations), split)
    manifest = json.loads((artifact / "exit_classifier.json").read_text())
    if manifest["dataset_sha256"] != dataset_fingerprint(conversations):
        raise ValueError("Exit training and evaluation benchmark snapshots differ.")
    with (artifact / "model.onnx").open("rb") as handle:
        artifact_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    settings = settings.model_copy(
        update={
            "exit_provider": "huggingface",
            "hf_exit_model": manifest["base_model"] + " (fine-tuned)",
            "hf_exit_revision": artifact_sha256,
            "hf_exit_local_path": artifact,
        }
    )
    rows = []
    for example in examples:
        started, error = time.perf_counter(), None
        try:
            predicted = exit_label(settings, example["history"], artifact_path=artifact)
        except Exception as exception:
            predicted, error = None, type(exception).__name__
        rows.append(
            {
                "conversation_id": example["conversation_id"],
                "turn_id": example["turn_id"],
                "expected": "end" if example["label"] == "end" else "do_not_end",
                "predicted": predicted,
                "error": error,
                "is_opening": example["is_opening"],
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
    report = {
        "provider": "huggingface",
        "task": "exit",
        "split": split,
        "exit_model": manifest["base_model"] + " (fine-tuned)",
        "exit_provider": "huggingface",
        "exit_revision": artifact_sha256,
        "artifact_sha256": artifact_sha256,
        "evaluation_mode": "historical_prefix",
        "metric_version": "failure-aware-v2",
        "dataset_version": settings.dataset_version,
        "dataset_sha256": dataset_fingerprint(conversations),
        "created_at": datetime.now().astimezone().isoformat(),
        "training_epoch": manifest["selected_epoch"],
        "all_turns": summarize(rows, ["do_not_end", "end"]),
        "candidate_responses": summarize(
            [row for row in rows if not row["is_opening"]], ["do_not_end", "end"]
        ),
        "rows": rows,
        "limitations": [
            "Three held-out conversations; results are exploratory.",
            "Test data was excluded from training and checkpoint/quantization selection.",
        ],
    }
    if output:
        from app.fine_tuning.huggingface import write_json

        write_json(output, report)
    return report
