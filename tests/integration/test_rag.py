import json
from unittest.mock import patch

import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from reportlab.pdfgen import canvas
from sqlalchemy import delete, inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.files import ORIGINAL_PDF_NAME
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.models import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID, DocumentVersion, Job
from app.ingestion.operations import operate, startup_index_ready
from app.ingestion.registry import JobRegistry
from app.rag.ingest import ingest_pdf
from app.rag.models import RagChunk
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
            assert session.get(Job, legacy_job).status == "active"
            assert session.get(RagChunk, (settings.index_collection, "legacy-chunk")) is not None
        assert extra.read_bytes().startswith(b"Not a PDF")
    finally:
        retriever.close()


def test_partial_update_preserves_previous_pdf_and_retry_repairs(settings, monkeypatch):
    registry = JobRegistry(settings)
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
    registry = JobRegistry(settings)
    try:
        assert settings.index_collection != changed.index_collection
        assert first["version_id"] != second["version_id"]
        assert registry.active_versions(DEFAULT_JOB_ID, settings.index_collection) == [
            first["version_id"]
        ]
        with Session(registry.engine) as session:
            assert session.get(DocumentVersion, second["version_id"]).embedding_dimensions == 512
    finally:
        registry.close()


def test_startup_reuses_ready_index_without_manifest_and_repairs_missing_chunk(settings):
    operate(settings, "bootstrap", if_needed=True)
    settings.job_manifest_path.unlink()
    registry = JobRegistry(settings)
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


def test_lease_prevents_overlap_and_releases_on_failure(settings):
    registry = JobRegistry(settings)
    try:
        with registry.lease():
            with pytest.raises(ValueError, match="Another ingestion"):
                with registry.lease():
                    pass
        with pytest.raises(RuntimeError), registry.lease():
            raise RuntimeError("injected failure")
        with registry.lease():
            pass
    finally:
        registry.close()


@pytest.fixture
def registry():
    registry = JobRegistry(Settings(_env_file=None))
    yield registry
    registry.close()


def stage(registry, version_id, *, document_id="document", collection="test"):
    registry.save_document(DEFAULT_JOB_ID, document_id, "example.pdf")
    registry.stage(
        {
            "document_version_id": version_id,
            "document_id": document_id,
            "pdf_path": "example.pdf",
            "pdf_sha256": "a" * 64,
            "text_sha256": version_id,
            "pipeline_version": "test",
            "embedding_model": "test",
            "collection_name": collection,
            "expected_chunk_count": 2,
            "status": "staging",
        }
    )


def test_fresh_database_has_22_tables_including_langgraph(registry):
    with PostgresSaver.from_conn_string(Settings(_env_file=None).postgres_dsn) as saver:
        saver.setup()
    tables = set(inspect(registry.engine).get_table_names())
    assert len(tables) == 22
    assert {
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "checkpoint_migrations",
    } <= tables
    assert {"job_locations", "ingestion_items", "document_indexes"}.isdisjoint(tables)


def test_database_enforces_complete_and_unique_active_versions(registry):
    stage(registry, "old")
    registry.publish("document", "test", "old", 2, 512)
    stage(registry, "new")
    with pytest.raises(IntegrityError), Session(registry.engine) as session, session.begin():
        session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_version_id == "new")
            .values(is_active=True)
        )
    stage(registry, "other-collection", collection="other")
    registry.publish("document", "other", "other-collection", 2, 512)
    registry.publish("document", "test", "new", 2, 512)
    with pytest.raises(IntegrityError), Session(registry.engine) as session, session.begin():
        session.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_version_id == "old")
            .values(is_active=True)
        )
    with Session(registry.engine) as session:
        assert set(
            session.scalars(
                select(DocumentVersion.document_version_id).where(DocumentVersion.is_active)
            )
        ) == {"new", "other-collection"}
