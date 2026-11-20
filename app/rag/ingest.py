"""Offline assignment PDF indexing, publication, readiness and bootstrap."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import select

from app.config import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID, ORIGINAL_PDF_NAME, ROOT
from app.observability.logging import configure_logging
from app.observability.tracing import TraceRecorder, traced
from app.rag.description import (
    FORMAT_VERSION,
    digest,
    normalized_text,
    pdf_markdown,
    presentation_for,
    validate_pdf_description,
)
from app.rag.models import RagChunk
from app.rag.repository import JobDescriptionRepository
from app.rag.retriever import vectorstore


def resolve_pdf(settings, value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    expected = settings.job_pdf_path.resolve() / ORIGINAL_PDF_NAME
    if path != expected:
        raise ValueError(f"Only data/jobs/{ORIGINAL_PDF_NAME} can be ingested")
    return path


def pdf_pages(path):
    pages = []
    for number, page in enumerate(PdfReader(path).pages, 1):
        if len(page.get_contents().get_data() if page.get_contents() else b"") > 20_000_000:
            raise ValueError("PDF page is too large to extract safely")
        text = normalized_text(page.extract_text() or "")
        if not text:
            raise ValueError(
                f"Page {number} has no extractable text; OCR is required before ingestion"
            )
        pages.append((number, text))
    if not pages:
        raise ValueError("PDF contains no pages")
    return pages


class DocumentIndexer:
    def __init__(self, settings, registry, store=None):
        self.settings, self.registry = settings, registry
        self.store = store if store is not None else vectorstore(settings, engine=registry.engine)
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    @traced("ingestion", "index_document")
    def index(self, *, job_id, document_id, pdf_path, source_url=""):
        if job_id != DEFAULT_JOB_ID or document_id != ASSIGNMENT_DOCUMENT_ID:
            raise ValueError("Only the assignment job description can be indexed")
        pdf_path = resolve_pdf(self.settings, pdf_path)
        if pdf_path.stat().st_size > 25_000_000:
            raise ValueError("Document exceeds the 25 MB ingestion limit")
        pages = pdf_pages(pdf_path)
        pdf_data = pdf_path.read_bytes()
        pdf_hash = digest(pdf_data)
        job = self.registry.get_job(job_id)
        presentation = self.registry.current_presentation(
            document_id, self.settings.index_collection, pdf_hash
        )
        if presentation is None:
            presentation = presentation_for(job, pdf_markdown(pdf_path, job["title"]))
        validate_pdf_description(presentation["description_text"], pages)
        text_hash = digest("\n".join(text for _, text in pages))
        version_id = digest(f"{document_id}:{text_hash}:{self.settings.index_collection}")
        documents = [Document(page_content=text, metadata={"page": page}) for page, text in pages]
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size, chunk_overlap=self.settings.chunk_overlap
        ).split_documents(documents)
        if not chunks:
            raise ValueError("No document chunks extracted")
        ids = []
        for index, chunk in enumerate(chunks):
            chunk_id = f"{version_id}:{index}"
            ids.append(chunk_id)
            chunk.metadata.update(
                {
                    "chunk_id": chunk_id,
                    "job_id": job_id,
                    "document_id": document_id,
                    "document_version_id": version_id,
                    "chunk_index": index,
                    "source": pdf_path.name,
                    "source_url": source_url,
                    "text_sha256": digest(chunk.page_content),
                }
            )
        processed = self.settings.job_data_path / "processed/jobs" / job_id / version_id
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "document.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "document_id": document_id,
                    "version_id": version_id,
                    "text_sha256": text_hash,
                    "collection": self.settings.index_collection,
                },
                indent=2,
            )
        )
        (processed / "chunks.jsonl").write_text(
            "\n".join(
                json.dumps(
                    {"text": chunk.page_content, "metadata": chunk.metadata}, ensure_ascii=False
                )
                for chunk in chunks
            )
            + "\n"
        )
        old_version = self.registry.current_version(document_id, self.settings.index_collection)
        existing = self.store.get(version_id=version_id)
        if old_version == version_id and set(existing["ids"]) == set(ids):
            changed = [
                (key, {**metadata, "source": pdf_path.name, "source_url": source_url})
                for key, metadata in zip(existing["ids"], existing["metadatas"], strict=True)
                if metadata.get("source") != pdf_path.name
                or metadata.get("source_url") != source_url
            ]
            if changed:
                # Filename changes need metadata updates, without another embedding request.
                self.store.update_metadata(
                    ids=[key for key, _ in changed],
                    metadatas=[metadata for _, metadata in changed],
                )
            self.registry.refresh_version_file(version_id, pdf_path, presentation=presentation)
            return {
                "status": "unchanged",
                "chunks": len(ids),
                "embedding_model": self.settings.embedding_model,
                "version_id": version_id,
            }
        # Unpublished chunks can be repaired on retry; the current PDF stays visible.
        for start in range(0, len(chunks), 32):
            self.store.add_documents(chunks[start : start + 32], ids=ids[start : start + 32])
        indexed = self.store.get(version_id=version_id, include=["embeddings"])
        if set(indexed["ids"]) != set(ids):
            raise ValueError("Indexed chunk IDs do not match the complete document")
        self.registry.publish(
            {
                "collection_name": self.settings.index_collection,
                "document_version_id": version_id,
                "pdf_path": str(pdf_path),
                "source_url": source_url,
                "pdf_sha256": pdf_hash,
                "pdf_data": pdf_data,
                "presentation": presentation,
                "chunk_count": len(ids),
                "embedding_dimensions": len(indexed["embeddings"][0]),
            }
        )
        # SQL publication happens first. Old vectors are invisible even if cleanup fails.
        if old_version and old_version != version_id:
            try:
                self.store.delete(version_id=old_version)
            except Exception:
                pass  # A subsequent explicit cleanup reconciles unreferenced vectors.
        return {
            "status": "indexed",
            "chunks": len(ids),
            "embedding_model": self.settings.embedding_model,
            "source_hash": digest(pdf_path.read_bytes()),
            "version_id": version_id,
        }


def ingest_pdf(path, settings, *, job_id=DEFAULT_JOB_ID, document_id=ASSIGNMENT_DOCUMENT_ID):
    if job_id != DEFAULT_JOB_ID or document_id != ASSIGNMENT_DOCUMENT_ID:
        raise ValueError("Only the assignment job description can be ingested")
    path = resolve_pdf(settings, path)
    settings.require_api_key()
    configure_logging(settings.log_level)
    registry = JobDescriptionRepository(settings)
    try:
        with (
            registry.index_lock(),
            TraceRecorder().trace(
                None, str(uuid4()), component="ingestion", operation="ingest_pdf"
            ),
        ):
            result = DocumentIndexer(settings, registry).index(
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
    registry = JobDescriptionRepository(settings)
    try:
        if command == "list-jobs":
            return registry.list_jobs(active_only=False)
        if command == "bootstrap":
            if if_needed and startup_index_ready(settings, registry):
                return {"status": "ready", "ingestion_skipped": True}
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
