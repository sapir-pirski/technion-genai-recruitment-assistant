import argparse
import json
from datetime import UTC, date, datetime, time
from pathlib import Path
from uuid import uuid4

from openai import OpenAIError

from app.config import ROOT, Settings
from app.observability.logging import configure_logging


def main():
    parser = argparse.ArgumentParser(description="Recruitment chatbot tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check configuration without exposing secrets")
    ingest = commands.add_parser("ingest", help="Embed every local PDF in data/jobs using OpenAI")
    ingest.add_argument("--pdf", type=Path, help="Ingest one PDF already inside jobs")
    ingest.add_argument("--job-id", default="assignment-python-developer")
    ingest.add_argument("--document-id")
    sync = commands.add_parser(
        "sync-jobs", help="Refresh selected PDFs, then ingest the jobs folder"
    )
    sync.add_argument(
        "--offline", action="store_true", help="Read local PDFs without collecting sources"
    )
    sync.add_argument(
        "--cleanup", action="store_true", help="Remove unreferenced vectors after publication"
    )
    sync.add_argument("--trigger", choices=["manual", "scheduled"], default="manual")
    sync.add_argument(
        "--limit", type=int, help="Maximum approved imported jobs; never discovers or replaces jobs"
    )
    sync.add_argument("--sources", type=Path)
    export = commands.add_parser(
        "export-corpus", help="Package current PDFs and their manifest for offline reconstruction"
    )
    export.add_argument("--output", type=Path, default=ROOT / "output/job-corpus.zip")
    commands.add_parser("list-jobs", help="Show job IDs, status, and scheduling capability")
    commands.add_parser("ingestion-status", help="Show recent synchronization results")
    commands.add_parser("index-status", help="Check indexed document completeness")
    bootstrap = commands.add_parser(
        "bootstrap", help="Initialize storage and ingest every local job PDF"
    )
    bootstrap.add_argument(
        "--if-needed",
        action="store_true",
        help="Reuse complete indexes when PDFs and manifest are unchanged",
    )
    preparation = commands.add_parser(
        "prepare-data", help="Export grouped splits and Exit Advisor JSONL"
    )
    preparation.add_argument(
        "--base-model", help="Exit training base model; defaults to Settings.chat_model"
    )
    commands.add_parser(
        "export-conversations", help="Regenerate the conversation JSON snapshot from PostgreSQL"
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
        command.add_argument("--job-id", required=True)
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
    chat.add_argument("--job-id", default="assignment-python-developer")
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
    training = commands.add_parser("fine-tune", help="Submit a paid OpenAI fine-tuning job")
    training.add_argument("--base-model", required=True)
    training.add_argument("--training", type=Path, default=ROOT / "data/processed/exit_train.jsonl")
    training.add_argument(
        "--validation", type=Path, default=ROOT / "data/processed/exit_validation.jsonl"
    )
    hf_training = commands.add_parser(
        "fine-tune-hf", help="Train ModernBERT locally on the stored benchmark and export ONNX"
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
    job = commands.add_parser("fine-tune-status")
    job.add_argument("job_id")
    args = parser.parse_args()
    settings = Settings()
    configure_logging(settings.log_level)
    try:
        if args.command == "doctor":
            from sqlalchemy.engine import make_url

            from app.profiles import document_converter

            result = {
                "provider": "openai",
                "openai_key_configured": bool(settings.openai_api_key.get_secret_value()),
                "chat_model": settings.chat_model,
                "helper_model": settings.helper_model,
                "judge_model": settings.judge_model,
                "exit_model": settings.model_for("exit"),
                "exit_provider": settings.exit_provider,
                "model_policy": settings.model_policy(),
                "exit_model_configured": settings.exit_provider == "huggingface"
                or bool(settings.exit_model),
                "database_driver": make_url(settings.database_url).drivername,
                "business_timezone": settings.business_timezone,
                "dataset_version": settings.dataset_version,
                "document_converter_ready": bool(document_converter()),
            }
        elif args.command == "ingest":
            if args.pdf:
                from app.rag.ingest import ingest_pdf

                result = ingest_pdf(
                    args.pdf, settings, job_id=args.job_id, document_id=args.document_id
                )
            else:
                from app.ingestion.sync import sync_jobs

                result = sync_jobs(settings, offline=True, cleanup=True)
                if result["counts"]["failed"]:
                    print(json.dumps(result, indent=2))
                    parser.exit(1, "PDF ingestion had failures; inspect ingestion-status.\n")
        elif args.command == "sync-jobs":
            import logging

            from app.ingestion.sync import sync_jobs

            logging.getLogger("httpx").setLevel(logging.WARNING)
            overrides = {}
            if args.limit is not None:
                if not 1 <= args.limit <= 10000:
                    raise ValueError("Use --limit between 1 and 10000")
                overrides["corpus_limit"] = args.limit
            if args.sources:
                overrides["job_sources_path"] = args.sources
            settings = settings.model_copy(update=overrides)
            result = sync_jobs(
                settings,
                offline=args.offline,
                trigger=args.trigger,
                cleanup=args.cleanup,
            )
            if result["counts"]["failed"]:
                print(json.dumps(result, indent=2))
                parser.exit(
                    1,
                    "Synchronization had failures; inspect ingestion-status and the run report.\n",
                )
        elif args.command == "export-corpus":
            from app.ingestion.export import export_corpus

            result = export_corpus(settings, args.output)
        elif args.command in ("list-jobs", "ingestion-status", "index-status", "bootstrap"):
            from app.ingestion.operations import operate

            result = operate(
                settings, args.command, if_needed=args.command == "bootstrap" and args.if_needed
            )
            if args.command == "index-status" and not result["complete"]:
                print(json.dumps(result, indent=2))
                parser.exit(
                    1,
                    "PDF indexes are pending or incomplete. Run ingest with a configured OpenAI API key.\n",
                )
            if args.command == "bootstrap" and result["counts"]["failed"]:
                print(json.dumps(result, indent=2))
                parser.exit(1, "PDF ingestion had failures; inspect ingestion-status.\n")
        elif args.command == "calendar":
            from app.calendar import manage_calendar

            result = manage_calendar(settings, args)
        elif args.command == "export-conversations":
            from app.ingestion.registry import JobRegistry
            from app.storage.conversation_repository import ConversationRepository

            registry = JobRegistry(settings)
            try:
                result = ConversationRepository(
                    registry.engine, settings.conversation_export_path
                ).export_json()
            finally:
                registry.close()
        elif args.command == "prepare-data":
            from app.evaluation.dataset import prepare_dataset

            result = prepare_dataset(
                settings, ROOT / "data/processed", exit_base_model=args.base_model
            )
        elif args.command == "evaluate":
            if args.exit_provider:
                settings = settings.model_copy(update={"exit_provider": args.exit_provider})
            if args.task == "workflow":
                from app.evaluation.workflow import evaluate_workflows

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
                result = {"report": str(destination), "provider": "openai", **report["all_turns"]}
        elif args.command == "fine-tune-hf":
            from app.fine_tuning.huggingface import export_onnx, train

            result = train(settings, args.output, epochs=args.epochs, seed=args.seed)
            result["export"] = export_onnx(settings, args.output)
        elif args.command == "evaluate-exit-hf":
            from app.evaluation.runner import evaluate_local_exit

            result = evaluate_local_exit(
                settings, args.artifact, split=args.split, output=args.output
            )
        elif args.command == "activate-exit-hf":
            from app.fine_tuning.huggingface import activate_local

            result = activate_local(settings)
        elif args.command in ("fine-tune", "fine-tune-status"):
            from app.fine_tuning.jobs import status, submit

            if args.command == "fine-tune":
                result = submit(settings, args.training, args.validation, args.base_model)
                destination = ROOT / "artifacts" / f"{result['job_id']}.json"
                destination.parent.mkdir(exist_ok=True)
                destination.write_text(json.dumps(result, indent=2))
            else:
                result = status(settings, args.job_id)
        else:
            from app.service import ChatService

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
