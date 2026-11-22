import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.ingestion.description import (
    clean_markdown,
    pdf_markdown,
    plain_markdown,
    presentation_for,
    validate_pdf_description,
)
from app.ingestion.normalize import digest, normalized_text
from app.observability.tracing import traced
from app.rag.retriever import vectorstore


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
    def __init__(self, settings, registry, store=None, *, lease_owner=None):
        self.settings, self.registry = settings, registry
        self.lease_owner = lease_owner
        self.store = store if store is not None else vectorstore(settings, engine=registry.engine)
        if settings.chunk_overlap >= settings.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    @traced("ingestion", "index_document")
    def index(self, *, job_id, document_id, pdf_path, source_url="", job_snapshot=None):
        pdf_path = Path(pdf_path).resolve()
        if pdf_path.parent != self.settings.job_pdf_path.resolve():
            raise ValueError("Place PDFs directly in the jobs folder before ingestion")
        if pdf_path.stat().st_size > 25_000_000:
            raise ValueError("Document exceeds the 25 MB ingestion limit")
        pages = pdf_pages(pdf_path)
        pdf_data = pdf_path.read_bytes()
        pdf_hash = digest(pdf_data)
        job = job_snapshot or self.registry.get_job(job_id)
        if job_snapshot is not None:
            markdown = (
                clean_markdown(job["description_text"], job["title"])
                if job.get("description_format") == "markdown"
                else plain_markdown(job["description_text"], job["title"])
            )
            presentation = presentation_for(job, markdown)
            validate_pdf_description(markdown, pages)
        else:
            presentation = self.registry.current_presentation(
                document_id, self.settings.index_collection, pdf_hash
            )
            if presentation is None:
                manifest = (
                    json.loads(self.settings.job_manifest_path.read_text())
                    if self.settings.job_manifest_path.is_file()
                    else {"jobs": []}
                )
                saved = next(
                    (
                        record
                        for record in manifest["jobs"]
                        if record["document_id"] == document_id
                        and record.get("pdf_sha256") == pdf_hash
                        and record.get("description_format") == "markdown"
                    ),
                    None,
                )
                markdown = (
                    saved["description_text"] if saved else pdf_markdown(pdf_path, job["title"])
                )
                presentation = presentation_for(
                    job, markdown, notices=saved.get("description_notices", []) if saved else []
                )
                validate_pdf_description(markdown, pages)
        text_hash = digest("\n".join(text for _, text in pages))
        version_id = digest(f"{document_id}:{text_hash}:{self.settings.index_collection}")
        self.registry.save_document(job_id, document_id, pdf_path, source_url)
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
        self.registry.stage(
            {
                "document_version_id": version_id,
                "document_id": document_id,
                "pdf_path": str(pdf_path),
                "pdf_sha256": pdf_hash,
                "text_sha256": text_hash,
                "pipeline_version": self.settings.pipeline_version,
                "embedding_model": self.settings.embedding_identity,
                "embedding_dimensions": None,
                "collection_name": self.settings.index_collection,
                "expected_chunk_count": len(ids),
                "indexed_chunk_count": 0,
                "status": "staging",
                "presentation": presentation,
                "pdf_data": pdf_data,
            }
        )
        try:
            # PostgreSQL upserts let retries repair partial versions.
            batch_size = 32
            for start in range(0, len(chunks), batch_size):
                if self.lease_owner:
                    self.registry.renew_lease(self.lease_owner)
                self.store.add_documents(
                    chunks[start : start + batch_size], ids=ids[start : start + batch_size]
                )
            indexed = self.store.get(version_id=version_id, include=["embeddings"])
            if set(indexed["ids"]) != set(ids):
                raise ValueError("Indexed chunk IDs do not match the complete document")
            dimensions = len(indexed["embeddings"][0])
            self.registry.publish(
                document_id,
                self.settings.index_collection,
                version_id,
                len(ids),
                dimensions,
                lease_owner=self.lease_owner,
            )
        except Exception as exc:
            self.registry.fail_version(version_id, exc)
            raise
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

    def cleanup(self):
        if self.lease_owner:
            self.registry.renew_lease(self.lease_owner)
        # The caller holds the catalog lease. Never remove a published version.
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from app.ingestion.models import DocumentVersion, Job, JobDocument

        with Session(self.registry.engine) as session:
            active = set(
                session.scalars(
                    select(DocumentVersion.document_version_id)
                    .join(JobDocument, DocumentVersion.document_id == JobDocument.document_id)
                    .join(Job, JobDocument.job_id == Job.job_id)
                    .where(
                        DocumentVersion.collection_name == self.settings.index_collection,
                        DocumentVersion.is_active.is_(True),
                        JobDocument.enabled.is_(True),
                        Job.status == "active",
                    )
                )
            )
        records = self.store.get(include=["metadatas"])
        stale = [
            key
            for key, meta in zip(records["ids"], records["metadatas"], strict=True)
            if meta.get("document_version_id") not in active
        ]
        for offset in range(0, len(stale), 500):
            self.store.delete(ids=stale[offset : offset + 500])
        return len(stale)
