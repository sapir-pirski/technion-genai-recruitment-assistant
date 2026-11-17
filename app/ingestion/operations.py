"""Check and initialize the assignment PDF's published vector index."""

import hashlib

from sqlalchemy import select

from app.ingestion.description import FORMAT_VERSION
from app.ingestion.files import ORIGINAL_PDF_NAME
from app.ingestion.models import DEFAULT_JOB_ID
from app.ingestion.registry import JobRegistry
from app.rag.models import RagChunk


def startup_index_ready(settings, registry):
    try:
        records = registry.manifest(settings)["jobs"]
        if len(records) != 1:
            return False
        row = records[0]
        path = settings.job_pdf_path / ORIGINAL_PDF_NAME
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        return (
            row["status"] == "active"
            and row["pdf_sha256"] == digest
            and row["collection"] == settings.index_collection
            and row["render_version"] == FORMAT_VERSION
            and index_status(settings, registry)["complete"]
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def operate(settings, command, *, if_needed=False):
    registry = JobRegistry(settings)
    try:
        if command == "list-jobs":
            return registry.list_jobs(active_only=False)
        if command == "bootstrap":
            if if_needed and startup_index_ready(settings, registry):
                return {"status": "ready", "ingestion_skipped": True}
            from app.rag.ingest import ingest_pdf

            return ingest_pdf(settings.job_pdf_path / ORIGINAL_PDF_NAME, settings)
        if command == "index-status":
            return index_status(settings, registry)
        raise ValueError("Unknown assignment index operation")
    finally:
        registry.close()


def index_status(settings, registry):
    checks = []
    for item in registry.manifest(settings)["jobs"]:
        with registry.engine.connect() as connection:
            actual = (
                set(
                    connection.scalars(
                        select(RagChunk.chunk_id).where(
                            RagChunk.collection_id == settings.index_collection,
                            RagChunk.job_id == DEFAULT_JOB_ID,
                            RagChunk.document_version_id == item["document_version_id"],
                        )
                    )
                )
                if item["document_version_id"]
                else set()
            )
        expected = {f"{item['document_version_id']}:{index}" for index in range(item["chunks"])}
        checks.append(
            {
                "job_id": DEFAULT_JOB_ID,
                "document_id": item["document_id"],
                "expected_chunks": item["chunks"],
                "actual_chunks": len(actual),
                "complete": item["status"] == "active" and bool(expected) and actual == expected,
            }
        )
    complete = len(checks) == 1 and checks[0]["complete"]
    return {
        "missing_job_indexes": [] if complete else [DEFAULT_JOB_ID],
        "collection": settings.index_collection,
        "documents": len(checks),
        "complete": complete,
        "checks": checks,
    }
