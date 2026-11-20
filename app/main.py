import argparse
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from uuid import uuid4

from openai import OpenAIError

from app.config import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID, ORIGINAL_PDF_NAME, ROOT, Settings
from app.observability.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Recruitment chatbot tools")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check configuration without exposing secrets")
    doctor.add_argument(
        "--verify-exit",
        action="store_true",
        help="Fetch if configured and verify the selected fine-tuned Exit artifact",
    )
    ingest = commands.add_parser("ingest", help="Embed the assignment Python Developer PDF")
    ingest.add_argument("--pdf", type=Path, help="Path to the assignment PDF in data/jobs")
    ingest.add_argument("--job-id", choices=[DEFAULT_JOB_ID], default=DEFAULT_JOB_ID)
    ingest.add_argument(
        "--document-id", choices=[ASSIGNMENT_DOCUMENT_ID], default=ASSIGNMENT_DOCUMENT_ID
    )
    commands.add_parser("list-jobs", help="Show the assignment role's status")
    commands.add_parser("index-status", help="Check the assignment PDF's vector index")
    bootstrap = commands.add_parser("bootstrap", help="Initialize and index the assignment PDF")
    bootstrap.add_argument(
        "--if-needed", action="store_true", help="Reuse a complete, unchanged index"
    )
    preparation = commands.add_parser(
        "prepare-data", help="Prepare splits from local benchmark JSON"
    )
    preparation.add_argument("--output", type=Path, default=ROOT / "data/processed")
    commands.add_parser(
        "export-conversations", help="Regenerate the conversation JSON snapshot from PostgreSQL"
    )
    legacy = commands.add_parser(
        "copy-legacy", help="Copy legacy chats, bookings and checkpoints into an empty core schema"
    )
    legacy.add_argument("--source-schema", default="public")
    legacy.add_argument(
        "--dry-run", action="store_true", help="Validate the copy, then roll it back"
    )
    calendar = commands.add_parser(
        "calendar", help="Manage recruiter-approved interview availability"
    )
    calendar_actions = calendar.add_subparsers(dest="calendar_action", required=True)
    status = calendar_actions.add_parser(
        "status", help="Show remaining availability and calendar expiration"
    )
    add = calendar_actions.add_parser(
        "add", help="Add an explicitly approved date range; existing slots are preserved"
    )
    for command in (status, add):
        command.add_argument("--job-id", choices=[DEFAULT_JOB_ID], default=DEFAULT_JOB_ID)
    status.add_argument("--timezone", help="Display timezone; defaults to the business timezone")
    add.add_argument("--from-date", required=True, type=date.fromisoformat)
    add.add_argument("--to-date", required=True, type=date.fromisoformat)
    add.add_argument(
        "--weekdays",
        required=True,
        nargs="+",
        choices=("mon", "tue", "wed", "thu", "fri", "sat", "sun"),
    )
    add.add_argument("--start-time", required=True, type=time.fromisoformat)
    add.add_argument("--end-time", required=True, type=time.fromisoformat)
    add.add_argument("--timezone", required=True, help="Recruiter calendar IANA timezone")
    add.add_argument("--duration-minutes", required=True, type=int)
    add.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without changing slots; diagnostics are still recorded",
    )
    chat = commands.add_parser("chat", help="Run a terminal conversation")
    chat.add_argument("--job-id", choices=[DEFAULT_JOB_ID], default=DEFAULT_JOB_ID)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument(
        "--split", choices=["train", "validation", "test", "all"], default="test"
    )
    evaluation.add_argument(
        "--task", choices=["actions", "exit", "quality", "workflow"], default="actions"
    )
    evaluation.add_argument(
        "--input", type=Path, help="Explicit quality packets or controlled workflow cases"
    )
    evaluation.add_argument(
        "--limit", type=int, default=20, help="Maximum quality cases, 1-200 (default: 20)"
    )
    evaluation.add_argument("--output", type=Path)
    evaluation.add_argument(
        "--exit-provider",
        choices=["openai", "huggingface"],
        help="Explicit Exit provider for reproducible baseline comparisons",
    )
    hf_training = commands.add_parser(
        "fine-tune-hf", help="Train ModernBERT locally from benchmark JSON and export ONNX"
    )
    hf_training.add_argument("--output", type=Path, default=ROOT / "artifacts/hf-exit")
    hf_training.add_argument("--epochs", type=int, choices=range(1, 21), default=5)
    hf_training.add_argument("--seed", type=int, default=42)
    hf_evaluate = commands.add_parser(
        "evaluate-exit-hf", help="Evaluate a local Exit artifact without uploading data"
    )
    hf_evaluate.add_argument("--artifact", type=Path, default=ROOT / "artifacts/hf-exit/runtime")
    hf_evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    hf_evaluate.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/exit_test_huggingface.json"
    )
    commands.add_parser(
        "activate-exit-hf", help="Activate the validated local Exit artifact on this machine only"
    )
    for command in (preparation, evaluation, hf_training, hf_evaluate):
        command.add_argument("--dataset", type=Path, help="Local benchmark JSON (or DATASET_PATH)")
        command.add_argument("--dataset-version", help="Benchmark version (or DATASET_VERSION)")
    args = parser.parse_args()
    dataset_settings = {}
    if getattr(args, "dataset", None) is not None:
        dataset_settings["dataset_path"] = args.dataset
    if getattr(args, "dataset_version", None) is not None:
        dataset_settings["dataset_version"] = args.dataset_version
    settings = Settings(**dataset_settings)
    configure_logging(settings.log_level)
    try:
        if args.command == "doctor":
            from sqlalchemy.engine import make_url

            result = {
                "provider": "openai",
                "openai_key_configured": bool(settings.openai_api_key.get_secret_value()),
                "chat_model": settings.chat_model,
                "helper_model": settings.helper_model,
                "judge_model": settings.judge_model,
                "exit_model": settings.model_for("exit"),
                "exit_provider": settings.exit_provider,
                "model_policy": settings.model_policy(),
                "database_driver": make_url(settings.database_url).drivername
                if settings.database_url
                else None,
                "business_timezone": settings.business_timezone,
                "database_schema": settings.database_schema,
                "dataset_version": settings.dataset_version,
            }
            if args.verify_exit:
                if settings.exit_provider != "huggingface":
                    parser.exit(
                        1,
                        "The configured Exit provider is a prompted baseline, not the fine-tuned model.\n",
                    )
                from app.agents.backends.huggingface import (
                    artifact_directory,
                    verify_local_artifact,
                )

                artifact = verify_local_artifact(
                    artifact_directory(settings), settings.hf_exit_revision
                )
                result.update(
                    exit_artifact_verified=True,
                    exit_artifact_sha256=artifact["files"]["model.onnx"],
                    exit_artifact_source="huggingface_hub" if settings.hf_exit_repo_id else "local",
                )
        elif args.command == "copy-legacy":
            from app.storage.legacy import copy_legacy

            result = copy_legacy(settings, source_schema=args.source_schema, dry_run=args.dry_run)
        elif args.command == "ingest":
            from app.rag.ingest import ingest_pdf

            result = ingest_pdf(args.pdf or settings.job_pdf_path / ORIGINAL_PDF_NAME, settings)
        elif args.command in ("list-jobs", "index-status", "bootstrap"):
            from app.rag.ingest import operate

            result = operate(
                settings, args.command, if_needed=args.command == "bootstrap" and args.if_needed
            )
            if args.command == "index-status" and not result["complete"]:
                print(json.dumps(result, indent=2))
                parser.exit(1, "The assignment PDF index is incomplete. Run ingest.\n")
        elif args.command == "calendar":
            from app.scheduling.calendar import manage_calendar

            result = manage_calendar(settings, args)
        elif args.command == "export-conversations":
            from app.rag.repository import JobDescriptionRepository
            from app.storage.conversations import ConversationRepository

            registry = JobDescriptionRepository(settings)
            try:
                result = ConversationRepository(
                    registry.engine, settings.conversation_export_path
                ).export_json()
            finally:
                registry.close()
        elif args.command == "prepare-data":
            from app.dataset import prepare_dataset

            result = prepare_dataset(settings, args.output)
        elif args.command == "evaluate":
            if args.exit_provider:
                settings = settings.model_copy(update={"exit_provider": args.exit_provider})
            if args.task == "workflow":
                from app.evaluation.replay import evaluate_workflows

                if args.input is None:
                    raise ValueError("Workflow evaluation requires --input with controlled cases")
                destination = args.output or ROOT / "artifacts/workflow_evaluation.json"
                report = evaluate_workflows(settings, args.input, output=destination)
                result = {"report": str(destination), **report["summary"]}
            elif args.task == "quality":
                from app.evaluation.judge import evaluate_quality

                if args.input is None:
                    raise ValueError(
                        "Quality evaluation requires --input with authorized, self-contained cases"
                    )
                destination = args.output or ROOT / "artifacts/quality_evaluation.json"
                report = evaluate_quality(
                    settings, args.input, output=destination, limit=args.limit
                )
                result = {
                    "report": str(destination),
                    "judge_model": report["judge_model"],
                    **report["summary"],
                    "usage": report["usage"],
                }
            else:
                from app.evaluation.runner import evaluate

                if args.input is not None:
                    raise ValueError("--input is used only with --task quality or workflow")
                destination = args.output or ROOT / f"artifacts/{args.task}_{args.split}.json"
                report = evaluate(settings, split=args.split, task=args.task, output=destination)
                result = {
                    "report": str(destination),
                    "provider": report["provider"],
                    **report["all_turns"],
                }
        elif args.command == "fine-tune-hf":
            from app.fine_tuning.train import export_onnx, train

            result = train(settings, args.output, epochs=args.epochs, seed=args.seed)
            result["export"] = export_onnx(settings, args.output)
        elif args.command == "evaluate-exit-hf":
            from app.evaluation.runner import evaluate_local_exit

            result = evaluate_local_exit(
                settings, args.artifact, split=args.split, output=args.output
            )
        elif args.command == "activate-exit-hf":
            from app.fine_tuning.train import activate_local

            result = activate_local()
        else:
            from app.workflow.service import ChatService

            service = ChatService(settings)
            conversation = str(uuid4())
            print(
                f"Models: chat={settings.chat_model}, helpers={settings.helper_model}, exit={settings.model_for('exit')}. Type /quit to leave."
            )
            reply = service.turn(conversation, job_id=args.job_id)
            print(reply.response)
            try:
                while reply.conversation_status != "closed":
                    text = input("You: ").strip()
                    if text == "/quit":
                        break
                    reply = service.turn(conversation, text, reference_time=datetime.now(UTC))
                    print(f"[{reply.action}] {reply.response}")
            finally:
                service.close()
            return
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
    except OpenAIError as exc:
        parser.exit(
            1,
            f"OpenAI request failed ({type(exc).__name__}). Check the API key, account access, model availability, and quota.\n",
        )


if __name__ == "__main__":
    main()
