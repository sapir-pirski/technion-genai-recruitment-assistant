from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.files import ORIGINAL_PDF_NAME, resolve_pdf
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.models import DEFAULT_JOB_ID, JobDocument
from app.ingestion.normalize import digest
from app.ingestion.registry import JobRegistry
from app.observability.logging import configure_logging
from app.observability.tracing import TraceRecorder


def ingest_pdf(path: Path, settings: Settings, *, job_id=DEFAULT_JOB_ID, document_id=None) -> dict:
    """Attach a PDF from the single input folder to an existing job."""
    path = resolve_pdf(settings, path)
    settings.require_api_key()
    configure_logging(settings.log_level)
    registry = JobRegistry(settings)
    try:
        if not registry.get_job(job_id):
            raise ValueError("Unknown job ID; register the job before attaching documents")
        if path.name == ORIGINAL_PDF_NAME and job_id == DEFAULT_JOB_ID:
            document_id = document_id or "assignment-job-description"
        if document_id and document_id.startswith("posting-"):
            raise ValueError("The posting- namespace is reserved for collected descriptions")
        if not document_id:
            with Session(registry.engine) as session:
                document_id = session.scalar(
                    select(JobDocument.document_id).where(
                        JobDocument.job_id == job_id, JobDocument.source_path == str(path)
                    )
                )
        document_id = document_id or "pdf-" + digest(f"{job_id}:{path}")[:32]
        if len(document_id) > 80:
            raise ValueError(
                "Use a document ID of at most 80 characters outside the reserved posting- namespace"
            )
        with (
            registry.lease() as owner,
            TraceRecorder(registry.engine).trace(
                None,
                str(uuid4()),
                component="ingestion",
                operation="ingest_pdf",
                attributes={"job_id": job_id},
            ),
        ):
            result = DocumentIndexer(settings, registry, lease_owner=owner).index(
                job_id=job_id,
                document_id=document_id,
                pdf_path=path,
                source_url=registry.get_job(job_id)["canonical_url"],
            )
            from app.ingestion.sync import atomic_json, manifest_paths

            atomic_json(settings.job_manifest_path, manifest_paths(registry.manifest(settings)))
            return result
    finally:
        registry.close()
