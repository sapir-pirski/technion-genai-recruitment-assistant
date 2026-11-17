"""Explicit offline ingestion of the assignment's Python Developer PDF."""

import json
from uuid import uuid4

from app.ingestion.files import ORIGINAL_PDF_NAME, resolve_pdf
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.models import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID
from app.ingestion.registry import JobRegistry
from app.observability.logging import configure_logging
from app.observability.tracing import TraceRecorder


def ingest_pdf(path, settings, *, job_id=DEFAULT_JOB_ID, document_id=ASSIGNMENT_DOCUMENT_ID):
    if job_id != DEFAULT_JOB_ID or document_id != ASSIGNMENT_DOCUMENT_ID:
        raise ValueError("Only the assignment job description can be ingested")
    path = resolve_pdf(settings, path)
    settings.require_api_key()
    configure_logging(settings.log_level)
    registry = JobRegistry(settings)
    try:
        with (
            registry.lease() as owner,
            TraceRecorder(registry.engine).trace(
                None, str(uuid4()), component="ingestion", operation="ingest_pdf"
            ),
        ):
            result = DocumentIndexer(settings, registry, lease_owner=owner).index(
                job_id=DEFAULT_JOB_ID,
                document_id=ASSIGNMENT_DOCUMENT_ID,
                pdf_path=path,
            )
            # This is an output summary, never a source for discovering/restoring jobs.
            destination = settings.job_manifest_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            manifest = registry.manifest(settings)
            for record in manifest["jobs"]:
                record["source_path"] = record["pdf_path"] = f"data/jobs/{ORIGINAL_PDF_NAME}"
            temporary = destination.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(destination)
            return result
    finally:
        registry.close()
