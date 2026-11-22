import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from io import BytesIO
from threading import Barrier
from unittest.mock import Mock, patch

import httpx
import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from PIL import Image
from pypdf import PdfReader
from sqlalchemy import delete, inspect, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings
from app.ingestion.indexer import DocumentIndexer
from app.ingestion.models import DEFAULT_JOB_ID, Company, DocumentVersion, IngestionRun, Job
from app.ingestion.normalize import digest
from app.ingestion.operations import operate, startup_index_ready
from app.ingestion.registry import JobRegistry
from app.ingestion.sync import sync_jobs, write_snapshot
from app.rag.ingest import ingest_pdf
from app.rag.models import RagChunk
from app.rag.retriever import JobRetriever


@pytest.fixture
def settings():
    return Settings(_env_file=None, request_timeout=1, max_retries=0)


def posting(
    job_id="one", text="Python REST services with PostgreSQL.", title="Python Backend Engineer"
):
    return {
        "job_id": job_id,
        "source_id": "greenhouse:example",
        "external_job_id": job_id,
        "canonical_url": f"https://example.com/jobs/{job_id}",
        "company_name": "Example",
        "title": title,
        "description_text": (text + "\n") * 15,
        "locations": [{"location_text": "London"}],
    }


def register(registry, job):
    registry.source({"provider": "greenhouse", "board": "example", "company": "Example"})
    registry.save_job(job)


def index_job(settings, registry, job, *, document_id="doc", store=None):
    pdf = write_snapshot(job, settings, previous=registry.get_job(job["job_id"]))
    register(registry, job)
    result = DocumentIndexer(settings, registry, store).index(
        job_id=job["job_id"],
        document_id=document_id,
        pdf_path=pdf,
        source_url=job["canonical_url"],
    )
    return result


def test_pdf_ingestion_is_idempotent_and_retrieves_sources():
    settings = Settings(_env_file=None)
    path = settings.job_pdf_path / "Python Developer Job Description.pdf"
    first = ingest_pdf(path, settings)
    second = ingest_pdf(path, settings)
    assert first["status"] == "indexed" and first["chunks"] > 1
    assert second["status"] == "unchanged" and first["chunks"] == second["chunks"]
    results = JobRetriever(settings).search("Python experience and required skills")
    assert results and all(item["id"] and item["page"] >= 1 for item in results)


def test_role_filter_never_returns_another_job(settings):
    registry = JobRegistry(settings)
    try:
        index_job(
            settings, registry, posting("one", "Python salary is GBP 90000."), document_id="one"
        )
        index_job(
            settings, registry, posting("two", "Python salary is USD 200000."), document_id="two"
        )
        retriever = JobRetriever(settings)
        try:
            matches = retriever.search("What is the salary?", job_id="one")
            assert matches and all(
                m["job_id"] == "one" and "200000" not in m["text"] for m in matches
            )
            assert not retriever.search("What is the salary?", job_id="unknown")
            with Session(registry.engine) as session, session.begin():
                session.get(Job, "one").status = "closed"
            assert not retriever.search("Python", job_id="one")
        finally:
            retriever.close()
    finally:
        registry.close()


def test_partial_update_preserves_previous_version_and_retry_repairs(settings, monkeypatch):
    registry = JobRegistry(settings)
    retriever = JobRetriever(settings)
    try:
        old = index_job(settings, registry, posting("one", "Original Python requirements."))
        published_pdf = registry.documents("one", collection=settings.index_collection)[0]["data"]
        store = retriever.store
        real_add = store.add_documents

        def fail_after_first(documents, ids):
            real_add(documents[:1], ids=ids[:1])
            raise TimeoutError("injected interruption")

        monkeypatch.setattr(store, "add_documents", fail_after_first)
        with pytest.raises(TimeoutError):
            index_job(
                settings, registry, posting("one", "Changed Python requirements."), store=store
            )
        assert registry.current_version("doc", settings.index_collection) == old["version_id"]
        assert (
            "Original"
            in registry.get_job("one", collection=settings.index_collection)["description_text"]
        )
        assert (
            "Changed"
            not in registry.get_job("one", collection=settings.index_collection)["description_text"]
        )
        assert (
            registry.documents("one", collection=settings.index_collection)[0]["data"]
            == published_pdf
        )
        assert all("Original" in m["text"] for m in retriever.search("Python", job_id="one"))
        monkeypatch.setattr(store, "add_documents", real_add)
        new = index_job(
            settings, registry, posting("one", "Changed Python requirements."), store=store
        )
        assert new["version_id"] != old["version_id"]
        assert all("Changed" in m["text"] for m in retriever.search("Python", job_id="one"))
        assert (
            "Changed"
            in registry.get_job("one", collection=settings.index_collection)["description_text"]
        )
        assert (
            registry.documents("one", collection=settings.index_collection)[0]["data"]
            != published_pdf
        )
        # Missing vectors cannot pass the unchanged check; the next run repairs them.
        key = store.get(version_id=new["version_id"])["ids"][0]
        store.delete(ids=[key])
        repaired = index_job(
            settings, registry, posting("one", "Changed Python requirements."), store=store
        )
        assert repaired["status"] == "indexed"
    finally:
        retriever.close()
        registry.close()


def test_configuration_change_requires_separate_index(settings):
    registry = JobRegistry(settings)
    try:
        first = index_job(settings, registry, posting())
        changed = settings.model_copy(update={"chunk_size": 500, "chunk_overlap": 50})
        second = index_job(changed, registry, posting())
        assert settings.index_collection != changed.index_collection
        assert first["version_id"] != second["version_id"]
        assert registry.active_versions("one", settings.index_collection) == [first["version_id"]]
        with Session(registry.engine) as session:
            assert session.get(DocumentVersion, second["version_id"]).embedding_dimensions == 512
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


def test_fresh_catalog_can_rebuild_captures_and_failed_rebuild_preserves_manifest(
    settings, monkeypatch, database_factory
):
    registry = JobRegistry(settings)
    try:
        job = posting()
        register(registry, job)
        pdf = write_snapshot(job, settings)
        registry.save_document("one", "posting-one", pdf, job["canonical_url"])
        manifest = registry.manifest(settings)
        assert manifest["jobs"][0]["index_status"] == "pending"
        assert manifest["jobs"][0]["document_version_id"] is None
        settings.job_manifest_path.write_text(json.dumps(manifest))
    finally:
        registry.close()
    fresh = settings.model_copy(
        update={
            "database_url": database_factory(),
        }
    )
    monkeypatch.setattr(
        httpx.Client, "get", lambda *args, **kwargs: pytest.fail("Network used in offline rebuild")
    )
    report = sync_jobs(fresh, offline=True)
    assert report["counts"]["failed"] == 0 and report["counts"]["added"] == 1
    assert report["catalog_jobs"] == 2
    repeated = sync_jobs(fresh, offline=True)
    assert repeated["counts"]["failed"] == 0
    assert repeated["counts"]["unchanged"] == 2
    # A clone without captured files must fail without erasing the acquisition list.
    manifest = json.loads(settings.job_manifest_path.read_text())
    imported = next(record for record in manifest["jobs"] if record["job_id"] == "one")
    imported["pdf_path"] = str(settings.job_pdf_path / "missing.pdf")
    original_import = next(p for p in settings.job_pdf_path.glob("*__*.pdf"))
    original_import.unlink()
    settings.job_manifest_path.write_text(json.dumps(manifest))
    empty = fresh.model_copy(
        update={
            "database_url": database_factory(),
        }
    )
    failed = sync_jobs(empty, offline=True)
    assert failed["counts"]["failed"] == 1
    saved = json.loads(settings.job_manifest_path.read_text())
    assert any(
        record["job_id"] == "one" and record.get("rebuild_status") == "pending"
        for record in saved["jobs"]
    )


def test_index_reads_edited_pdf_instead_of_saved_source_text(settings):
    from app.ingestion.pdf_snapshot import make_snapshot

    registry = JobRegistry(settings)
    try:
        index_job(settings, registry, posting(text="Original source requirements."))
        manifest = registry.manifest(settings)
        settings.job_manifest_path.write_text(json.dumps(manifest))
        path = next(settings.job_pdf_path.glob("*__*.pdf"))
        make_snapshot(
            posting(text="Updated PDF requirements with FastAPI."), path, "2026-11-02T12:00:00Z"
        )
        report = sync_jobs(settings, offline=True, cleanup=True)
        assert report["counts"]["failed"] == 0 and report["counts"]["updated"] >= 1
        retriever = JobRetriever(settings)
        try:
            matches = retriever.search("FastAPI requirements", job_id="one")
            assert matches and any("Updated PDF requirements" in match["text"] for match in matches)
            assert all(match["page"] >= 1 for match in matches)
        finally:
            retriever.close()
        assert not (settings.job_data_path / "raw").exists()
        published = registry.get_job("one", collection=settings.index_collection)
        assert "Updated PDF requirements with FastAPI" in published["description_text"]
        assert "Original source requirements" not in published["description_text"]
        assert (
            registry.documents("one", collection=settings.index_collection)[0]["data"]
            == path.read_bytes()
        )
    finally:
        registry.close()


def test_new_pdf_rename_and_deletion_preserve_identity_and_remove_vectors(settings):
    from app.ingestion.pdf_snapshot import make_snapshot

    path = settings.job_pdf_path / "new-role.pdf"
    make_snapshot(posting(text="New local PDF requirements."), path, "2026-11-02T12:00:00Z")
    first = sync_jobs(settings, offline=True, cleanup=True)
    assert first["counts"]["failed"] == 0 and first["counts"]["added"] == 1
    registry = JobRegistry(settings)
    try:
        job_id = next(
            job["job_id"] for job in registry.list_jobs() if job["job_id"] != DEFAULT_JOB_ID
        )
        before = registry.active_versions(job_id, settings.index_collection)
        renamed = path.with_name("renamed-role.pdf")
        path.replace(renamed)
        second = sync_jobs(settings, offline=True, cleanup=True)
        assert second["counts"]["failed"] == 0 and second["counts"]["added"] == 0
        assert registry.active_versions(job_id, settings.index_collection) == before
        renamed.unlink()
        third = sync_jobs(settings, offline=True, cleanup=True)
        assert third["counts"]["removed_documents"] == 1
        assert registry.get_job(job_id)["status"] == "removed"
        assert not DocumentIndexer(settings, registry).store.get(job_id=job_id)["ids"]
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


def test_concurrent_run_items_are_not_lost_or_overwritten_by_finish(registry):
    run_id = registry.start_run("test")
    barrier = Barrier(8)

    def record(index):
        barrier.wait(timeout=10)
        registry.record_item(
            run_id, "index", job_id=f"job-{index}", error=TimeoutError() if index == 0 else None
        )
        if index == 7:
            registry.finish_run(run_id, {"indexed": 31, "failed": 1})

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(32)))
    with Session(registry.engine) as session:
        run = session.get(IngestionRun, run_id)
        assert len(run.items) == len({item["item_id"] for item in run.items}) == 32
        assert {item["job_id"] for item in run.items} == {f"job-{i}" for i in range(32)}
        failed = [item for item in run.items if item["status"] == "failed"]
        assert len(failed) == 1 and failed[0]["error_code"] == "TimeoutError"
        assert run.status == "partial" and run.counts == {"indexed": 31, "failed": 1}
    with pytest.raises(ValueError, match="Unknown ingestion run"):
        registry.record_item("missing", "index")


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


def test_reviewed_collection_preserves_job_scope_and_existing_data():
    settings = Settings(_env_file=None, request_timeout=1, max_retries=0)
    job_id = "job-" + digest("web:example:selected")[:24]
    source = {
        "provider": "web",
        "board": "example",
        "company": "Example",
        "postings": [
            {
                "url": "https://example.com/jobs/selected",
                "external_job_id": "selected",
                "title": "Python Developer",
                "start_marker": "Python Developer",
                "end_marker": "Next Job",
                "locations": ["London"],
                "workplace_type": "hybrid",
            }
        ],
        "selected_job_ids": [job_id],
    }
    settings.job_sources_path.write_text(
        json.dumps({"version": 1, "selection_mode": "reviewed_only", "sources": [source]})
    )
    body = (
        "<h1>Other Job</h1><p>Java neighbor</p><h1>Python Developer</h1>"
        "<p>Develop Python applications for internal document management. Maintain existing services "
        "and collaborate with product teams. Build API integrations, improve reliability, and "
        "document operational procedures for our customers and support teams.</p>"
        "<h1>Next Job</h1><p>React neighbor</p>"
    )
    registry = JobRegistry(settings)
    try:
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=body))
        ) as client:
            result = sync_jobs(settings, client=client)
            assert result["counts"]["failed"] == 0
            assert registry.active_versions(job_id, settings.index_collection)
            listed = next(job for job in registry.list_jobs() if job["job_id"] == job_id)
            assert listed["locations"] == ["London"] and listed["workplace_type"] == "hybrid"
            pdf = next(settings.job_pdf_path.glob("*__*.pdf"))
            text = "\n".join(page.extract_text() for page in PdfReader(pdf).pages)
            assert "Develop Python applications" in text
            assert "Java neighbor" not in text and "React neighbor" not in text
            assert source["postings"][0]["url"] in text
            before = pdf.read_bytes()
            repeated = sync_jobs(settings, client=client)
            assert repeated["counts"]["failed"] == 0 and repeated["counts"]["unchanged"] == 2
            assert pdf.read_bytes() == before
            assert registry.get_job(job_id)["workplace_type"] == "hybrid"
        with httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(503))
        ) as client:
            failed = sync_jobs(settings, client=client)
        assert failed["counts"]["failed"] == 1
        assert registry.get_job(job_id)["status"] == "active"
        assert registry.get_job(job_id)["missing_scans"] == 0
        assert pdf.read_bytes() == before
        assert not (settings.job_data_path / "raw").exists()
    finally:
        registry.close()


def test_company_logo_refresh_is_independent_of_pdf_and_embeddings():
    settings = Settings(_env_file=None, max_retries=0, request_timeout=1)
    job_id = "job-" + digest("web:example:selected")[:24]
    source = {
        "provider": "web",
        "board": "example",
        "company": "Example",
        "selected_job_ids": [job_id],
        "postings": [
            {
                "url": "https://company.example/careers",
                "external_job_id": "selected",
                "title": "Python Developer",
                "start_marker": "Python Developer",
                "end_marker": "Next Job",
            }
        ],
    }
    settings.job_sources_path.write_text(
        json.dumps({"version": 1, "selection_mode": "reviewed_only", "sources": [source]})
    )
    body = (
        '<header><img src="/logo.png" alt="Example logo"></header>'
        "<h1>Python Developer</h1><p>"
        + "Build Python APIs and work with PostgreSQL. " * 15
        + "</p><h1>Next Job</h1>"
    )
    mode = {"value": "red"}
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.path == "/logo.png":
            if mode["value"] == "304":
                assert request.headers["if-none-match"] == '"blue"'
                return httpx.Response(304)
            if mode["value"] == "error":
                return httpx.Response(503)
            image = BytesIO()
            Image.new("RGB", (32, 32), mode["value"]).save(image, "PNG")
            return httpx.Response(
                200, content=image.getvalue(), headers={"etag": f'"{mode["value"]}"'}
            )
        return httpx.Response(200, text=body)

    registry = JobRegistry(settings)
    try:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            first = sync_jobs(settings, client=client)
            assert first["counts"]["logos_updated"] == 1
            company_id = registry.get_job(job_id)["company_id"]
            with Session(registry.engine) as session:
                company = session.get(Company, company_id)
                red_sha = company.logo_sha256
                assert company.logo_data.startswith(b"\x89PNG")
            pdf = next(settings.job_pdf_path.glob("*__*.pdf"))
            original_pdf = pdf.read_bytes()

            def vector_hash():
                with registry.engine.connect() as connection:
                    return connection.scalar(
                        text(
                            "SELECT md5(string_agg(chunk_id || text || embedding::text, '' ORDER BY chunk_id)) FROM rag_chunks"
                        )
                    )

            original_vectors = vector_hash()

            def due():
                with Session(registry.engine) as session, session.begin():
                    session.get(Company, company_id).logo_checked_at -= timedelta(days=1)

            due()
            mode["value"] = "blue"
            changed = sync_jobs(settings, client=client)
            assert changed["counts"]["logos_updated"] == 1
            assert changed["counts"]["unchanged"] == 2
            with Session(registry.engine) as session:
                company = session.get(Company, company_id)
                assert company.logo_sha256 != red_sha
                blue_sha, updated_at = company.logo_sha256, company.logo_updated_at
            due()
            mode["value"] = "304"
            repeated = sync_jobs(settings, client=client)
            assert repeated["counts"]["logos_unchanged"] == 1
            due()
            mode["value"] = "error"
            failed = sync_jobs(settings, client=client)
            assert failed["counts"]["failed"] == 0 and failed["counts"]["logos_failed"] == 1
            with Session(registry.engine) as session:
                company = session.get(Company, company_id)
                assert company.logo_sha256 == blue_sha and company.logo_status == "stale"
                assert company.logo_updated_at == updated_at
            assert pdf.read_bytes() == original_pdf and vector_hash() == original_vectors
            assert registry.get_job(job_id)["status"] == "active"
    finally:
        registry.close()


def test_startup_reuses_ready_indexes_and_detects_a_missing_chunk():
    settings = Settings()
    assert sync_jobs(settings, offline=True)["counts"]["failed"] == 0
    registry = JobRegistry(settings)
    try:
        assert startup_index_ready(settings, registry)
        with patch(
            "app.ingestion.sync.sync_jobs", side_effect=AssertionError("No ingestion needed")
        ):
            result = operate(settings, "bootstrap", if_needed=True)
        assert result["ingestion_skipped"]
        with registry.engine.begin() as connection:
            key = connection.scalar(select(RagChunk.chunk_id).limit(1))
            connection.execute(delete(RagChunk).where(RagChunk.chunk_id == key))
        assert not startup_index_ready(settings, registry)
        replacement = Mock(return_value={"counts": {"failed": 0}})
        with patch("app.ingestion.sync.sync_jobs", replacement):
            operate(settings, "bootstrap", if_needed=True)
        replacement.assert_called_once()
    finally:
        registry.close()
