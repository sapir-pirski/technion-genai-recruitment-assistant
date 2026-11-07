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
        "provider": "openai",
        "model": settings.chat_model,
        "helper_model": settings.helper_model,
        "exit_model": settings.model_for("exit"),
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
