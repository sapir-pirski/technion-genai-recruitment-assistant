import json
from unittest.mock import patch

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from reportlab.pdfgen import canvas
from sqlalchemy import delete, inspect, select, text
from sqlalchemy.orm import Session

from app.config import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID, ORIGINAL_PDF_NAME, Settings
from app.rag.ingest import DocumentIndexer, ingest_pdf, operate, startup_index_ready
from app.rag.models import JobDescription, RagChunk
from app.rag.repository import JobDescriptionRepository
from app.rag.retriever import JobRetriever


@pytest.fixture
def settings():
    return Settings(_env_file=None, request_timeout=1, max_retries=0)


def rewrite_assignment(settings, text):
    path = settings.job_pdf_path / ORIGINAL_PDF_NAME
    pdf = canvas.Canvas(str(path))
    for index in range(18):
        pdf.drawString(40, 800 - 24 * index, text)
    pdf.save()
    return path


def test_pdf_ingestion_is_idempotent_and_retrieves_sources(settings):
    path = settings.job_pdf_path / ORIGINAL_PDF_NAME
    first = ingest_pdf(path, settings)
    second = ingest_pdf(path, settings)
    assert first["status"] == "indexed" and first["chunks"] > 1
    assert second["status"] == "unchanged" and first["chunks"] == second["chunks"]
    retriever = JobRetriever(settings)
    try:
        results = retriever.search("Python experience and required skills")
        assert results and all(
            item["id"] and item["page"] >= 1 and item["job_id"] == DEFAULT_JOB_ID
            for item in results
        )
        assert retriever.references([results[0]["id"]], DEFAULT_JOB_ID)
    finally:
        retriever.close()


def test_extra_pdfs_and_legacy_jobs_are_ignored_without_deleting_stored_data(settings, legacy_job):
    extra = settings.job_pdf_path / "unrelated-role.pdf"
    extra.write_bytes(b"Not a PDF: startup must never read this file")
    operate(settings, "bootstrap", if_needed=True)
    retriever = JobRetriever(settings)
    registry = retriever.registry
    try:
        with Session(registry.engine) as session, session.begin():
            session.add(
                RagChunk(
                    collection_id=settings.index_collection,
                    chunk_id="legacy-chunk",
                    job_id=legacy_job,
                    text="Legacy-only salary USD 200000.",
                    embedding=[1.0] + [0.0] * 511,
                    dimensions=512,
                    attributes={"source": "legacy.pdf", "page": 1},
                )
            )
        assert startup_index_ready(settings, registry)
        assert [job["job_id"] for job in registry.list_jobs()] == [DEFAULT_JOB_ID]
        assert json.loads(json.dumps(operate(settings, "list-jobs")))[0]["job_id"] == DEFAULT_JOB_ID
        assert registry.get_job(legacy_job) is None and registry.documents(legacy_job) == []
        assert [row["job_id"] for row in registry.manifest(settings)["jobs"]] == [DEFAULT_JOB_ID]
        assert retriever.search("salary", job_id=legacy_job) == []
        assert retriever.references(["legacy-chunk"], legacy_job) == []
        matches = retriever.search("Python")
        assert matches and all("200000" not in match["text"] for match in matches)
        operate(settings, "bootstrap")
        with Session(registry.engine) as session:
            assert (
                session.execute(
                    text("SELECT status FROM public.jobs WHERE job_id = :job"), {"job": legacy_job}
                ).scalar_one()
                == "active"
            )
            assert session.get(RagChunk, (settings.index_collection, "legacy-chunk")) is not None
        assert extra.read_bytes().startswith(b"Not a PDF")
    finally:
        retriever.close()


def test_partial_update_preserves_previous_pdf_and_retry_repairs(settings, monkeypatch):
    registry = JobDescriptionRepository(settings)
    retriever = JobRetriever(settings)
    try:
        path = rewrite_assignment(settings, "Original Python requirements include SQL and APIs.")
        old = ingest_pdf(path, settings)
        published_pdf = registry.documents(DEFAULT_JOB_ID, collection=settings.index_collection)[0][
            "data"
        ]
        store = retriever.store
        real_add = store.add_documents

        def fail_after_first(documents, ids):
            real_add(documents[:1], ids=ids[:1])
            raise TimeoutError("injected interruption")

        path = rewrite_assignment(settings, "Changed Python requirements include FastAPI and SQL.")
        monkeypatch.setattr(store, "add_documents", fail_after_first)
        indexer = DocumentIndexer(settings, registry, store)
        with pytest.raises(TimeoutError):
            indexer.index(job_id=DEFAULT_JOB_ID, document_id=ASSIGNMENT_DOCUMENT_ID, pdf_path=path)
        assert (
            registry.current_version(ASSIGNMENT_DOCUMENT_ID, settings.index_collection)
            == old["version_id"]
        )
        assert (
            "Original"
            in registry.get_job(DEFAULT_JOB_ID, collection=settings.index_collection)[
                "description_text"
            ]
        )
        assert (
            registry.documents(DEFAULT_JOB_ID, collection=settings.index_collection)[0]["data"]
            == published_pdf
        )
        assert all("Original" in match["text"] for match in retriever.search("Python"))
        monkeypatch.setattr(store, "add_documents", real_add)
        new = indexer.index(
            job_id=DEFAULT_JOB_ID, document_id=ASSIGNMENT_DOCUMENT_ID, pdf_path=path
        )
        assert new["version_id"] != old["version_id"]
        assert all("Changed" in match["text"] for match in retriever.search("Python"))
        key = store.get(version_id=new["version_id"])["ids"][0]
        store.delete(ids=[key])
        repaired = indexer.index(
            job_id=DEFAULT_JOB_ID, document_id=ASSIGNMENT_DOCUMENT_ID, pdf_path=path
        )
        assert repaired["status"] == "indexed"
    finally:
        retriever.close()
        registry.close()


def test_configuration_change_requires_separate_index(settings):
    path = settings.job_pdf_path / ORIGINAL_PDF_NAME
    first = ingest_pdf(path, settings)
    changed = settings.model_copy(update={"chunk_size": 500, "chunk_overlap": 50})
    second = ingest_pdf(path, changed)
    registry = JobDescriptionRepository(settings)
    try:
        assert settings.index_collection != changed.index_collection
        assert first["version_id"] != second["version_id"]
        assert registry.active_versions(DEFAULT_JOB_ID, settings.index_collection) == [
            first["version_id"]
        ]
        with Session(registry.engine) as session:
            assert session.get(JobDescription, changed.index_collection).embedding_dimensions == 512
    finally:
        registry.close()


def test_startup_reuses_ready_index_without_manifest_and_repairs_missing_chunk(settings):
    operate(settings, "bootstrap", if_needed=True)
    settings.job_manifest_path.unlink()
    registry = JobDescriptionRepository(settings)
    try:
        assert startup_index_ready(settings, registry)
        with patch("app.rag.ingest.ingest_pdf", side_effect=AssertionError("No ingestion needed")):
            assert operate(settings, "bootstrap", if_needed=True)["ingestion_skipped"]
        with registry.engine.begin() as connection:
            key = connection.scalar(select(RagChunk.chunk_id).limit(1))
            connection.execute(delete(RagChunk).where(RagChunk.chunk_id == key))
        assert not startup_index_ready(settings, registry)
        operate(settings, "bootstrap", if_needed=True)
        assert startup_index_ready(settings, registry)
        path = settings.job_pdf_path / ORIGINAL_PDF_NAME
        path.unlink()
        assert not startup_index_ready(settings, registry)
        with pytest.raises(FileNotFoundError):
            operate(settings, "bootstrap", if_needed=True)
    finally:
        registry.close()


def test_advisory_lock_prevents_overlap_and_releases_on_failure(settings):
    registry = JobDescriptionRepository(settings)
    try:
        with registry.index_lock():
            with pytest.raises(ValueError, match="Another ingestion"):
                with registry.index_lock():
                    pass
        with pytest.raises(RuntimeError), registry.index_lock():
            raise RuntimeError("injected failure")
        with registry.index_lock():
            pass
    finally:
        registry.close()


@pytest.fixture
def registry():
    registry = JobDescriptionRepository(Settings(_env_file=None))
    yield registry
    registry.close()


def test_fresh_database_has_eleven_core_and_langgraph_tables(registry):
    settings = Settings(_env_file=None)
    with PostgresSaver.from_conn_string(settings.postgres_dsn) as saver:
        saver.setup()
    tables = set(inspect(registry.engine).get_table_names(schema=settings.database_schema))
    assert tables == {
        "interview_slots",
        "interview_bookings",
        "conversations",
        "conversation_messages",
        "job_descriptions",
        "rag_collections",
        "rag_chunks",
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    }
    assert inspect(registry.engine).get_table_names(schema="public") == []


def test_publication_rejects_missing_chunks_and_keeps_previous_pdf(registry):
    settings = Settings(_env_file=None)
    original = ingest_pdf(settings.job_pdf_path / ORIGINAL_PDF_NAME, settings)
    values = {
        "collection_name": settings.index_collection,
        "document_version_id": "missing-version",
        "pdf_path": "missing.pdf",
        "source_url": "",
        "pdf_sha256": "a" * 64,
        "pdf_data": b"missing",
        "presentation": {"description_text": "Unpublished change"},
        "chunk_count": 1,
        "embedding_dimensions": 512,
    }
    with pytest.raises(ValueError, match="Incomplete document index"):
        registry.publish(values)
    assert (
        registry.current_version(ASSIGNMENT_DOCUMENT_ID, settings.index_collection)
        == original["version_id"]
    )
