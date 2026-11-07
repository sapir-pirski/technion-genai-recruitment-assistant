"""Collect selected postings into PDFs, then ingest the single local PDF folder."""

import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import ROOT
from app.ingestion.description import (
    FORMAT_VERSION,
    clean_markdown,
    plain_markdown,
)
from app.ingestion.files import ORIGINAL_PDF_NAME, job_pdf_filename, resolve_pdf
from app.ingestion.indexer import DocumentIndexer, pdf_pages
from app.ingestion.logos import refresh_company_logos
from app.ingestion.models import DEFAULT_JOB_ID, DocumentVersion, Job, JobDocument, JobSource
from app.ingestion.normalize import digest
from app.ingestion.pdf_snapshot import make_snapshot
from app.ingestion.registry import JobRegistry, utcnow
from app.ingestion.sources.boards import fetch_board, load_sources, selected_job_ids
from app.ingestion.sources.reviewed import fetch_reviewed
from app.observability.logging import configure_logging, log_event
from app.observability.tracing import TraceRecorder


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def manifest_paths(value):
    for record in value["jobs"]:
        for name in ("pdf_path", "source_path"):
            try:
                record[name] = str(Path(record[name]).resolve().relative_to(ROOT))
            except ValueError:
                pass
    return value


def _source_identity(job):
    locations = [
        loc if isinstance(loc, str) else loc["location_text"] for loc in job.get("locations", [])
    ]
    description = (
        clean_markdown(job["description_text"], job["title"])
        if job.get("description_format") == "markdown"
        else plain_markdown(job["description_text"], job["title"])
    )
    return (
        job["title"],
        job["company_name"],
        job["canonical_url"],
        description,
        sorted(locations),
        job.get("workplace_type"),
        job.get("employment_type"),
        job.get("seniority"),
        json.dumps(job.get("compensation"), sort_keys=True),
        job.get("source_published_at"),
        tuple(job.get("description_notices", [])),
    )


def write_snapshot(job, settings, *, previous=None):
    """Write changed source content to its sole PDF, without raw sidecars."""
    path = settings.job_pdf_path / job_pdf_filename(job, settings)
    if job.get("original_pdf") is not None:
        content = job["original_pdf"]
        if path.is_file() and path.read_bytes() == content:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_bytes(content)
            pdf_pages(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
    if (
        path.exists()
        and previous
        and previous.get("render_version") == FORMAT_VERSION
        and _source_identity(previous) == _source_identity(job)
    ):
        return path
    if settings.job_manifest_path.exists():
        for record in json.loads(settings.job_manifest_path.read_text())["jobs"]:
            if Path(record["pdf_path"]).name == path.name and record["job_id"] != job["job_id"]:
                raise ValueError("Generated PDF filename collision")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    try:
        make_snapshot(job, temporary, datetime.now(UTC).isoformat())
        pdf_pages(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _restore_metadata(settings, registry, manifest, run_id, counts, *, require_files):
    created, missing = set(), set()
    allowed = selected_job_ids(settings)
    for record in manifest["jobs"]:
        try:
            path = resolve_pdf(settings, record["pdf_path"])
            if (
                not record["source_id"].startswith("local:")
                and allowed is not None
                and record["job_id"] not in allowed
            ):
                with Session(registry.engine) as session, session.begin():
                    existing = session.get(Job, record["job_id"])
                    if existing:
                        existing.status = "excluded"
                raise ValueError("Unapproved remote job in the corpus manifest")
            if not registry.get_job(record["job_id"]):
                provider, board = record["source_id"].split(":", 1)
                registry.source(
                    {
                        "provider": provider,
                        "board": board,
                        "company": record["company_name"],
                        "website_url": record.get("company_website_url"),
                    }
                )
                registry.save_job(
                    {
                        **record,
                        "description_text": record.get("description_text", ""),
                        "locations": record.get("locations", []),
                    }
                )
                created.add(record["job_id"])
                with Session(registry.engine) as session, session.begin():
                    session.get(Job, record["job_id"]).status = record.get("status", "active")
                if require_files and not path.exists():
                    missing.add(record["document_id"])
                    raise FileNotFoundError("Manifest PDF is missing during catalog reconstruction")
            registry.save_document(
                record["job_id"], record["document_id"], path, record["canonical_url"]
            )
        except Exception as exc:
            missing.add(record["document_id"])
            counts["failed"] += 1
            registry.record_item(run_id, "restore_metadata", job_id=record["job_id"], error=exc)
            log_event(
                "metadata_restore_failed",
                level=logging.WARNING,
                component="app.ingestion",
                job_id=record["job_id"],
                run_id=run_id,
                error_code=type(exc).__name__,
            )
    return created, missing


def _collect(settings, registry, run_id, counts, owner, client):
    configured = load_sources(settings.job_sources_path, include_disabled=True)
    for source in configured:
        registry.source(source)
    sources = [source for source in configured if source.get("enabled", True)]
    approved = {job_id for source in sources for job_id in source.get("selected_job_ids", [])}
    if len(approved) > settings.corpus_limit:
        raise ValueError("The explicit source selection exceeds CORPUS_LIMIT")
    allowed = selected_job_ids(settings)
    if allowed is not None:
        with Session(registry.engine) as session, session.begin():
            for job in session.scalars(select(Job).where(Job.status == "active")):
                if not job.source_id.startswith("local:") and job.job_id not in allowed:
                    job.status = "excluded"
    results, failed_sources, created = {}, set(), set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {
            pool.submit(
                fetch_reviewed if s["provider"] == "web" else fetch_board,
                s,
                client,
                retries=settings.max_retries,
            ): s
            for s in sources
        }
        for future in as_completed(pending):
            source = pending[future]
            key = f"{source['provider']}:{source['board']}"
            registry.renew_lease(owner)
            try:
                results[key] = future.result()
            except Exception as exc:
                failed_sources.add(key)
                counts["failed"] += 1
                registry.record_item(run_id, "fetch_source", error=exc)
                log_event(
                    "source_fetch_failed",
                    level=logging.WARNING,
                    component="app.ingestion",
                    source_id=key,
                    run_id=run_id,
                    error_code=type(exc).__name__,
                )
    candidates = {}
    for key, result in results.items():
        closed = registry.reconcile_source(key, result.all_ids)
        counts["closed"] += len(closed)
        for job_id in closed:
            registry.record_item(run_id, "closed", job_id=job_id)
        for job in result.jobs:
            if job["job_id"] not in approved:
                continue
            counts["selected"] += 1
            registry.renew_lease(owner)
            try:
                path = write_snapshot(job, settings, previous=registry.get_job(job["job_id"]))
                if registry.save_job(job):
                    created.add(job["job_id"])
                registry.save_document(
                    job["job_id"],
                    "posting-" + digest(job["job_id"])[:32],
                    path.resolve(),
                    job["canonical_url"],
                )
                candidates[job["job_id"]] = job
            except Exception as exc:
                failed_sources.add(key)
                counts["failed"] += 1
                registry.record_item(run_id, "collect_pdf", job_id=job["job_id"], error=exc)
                log_event(
                    "pdf_collection_failed",
                    level=logging.WARNING,
                    component="app.ingestion",
                    job_id=job["job_id"],
                    run_id=run_id,
                    error_code=type(exc).__name__,
                )
    with Session(registry.engine) as session, session.begin():
        for key in results.keys() - failed_sources:
            session.get(JobSource, key).last_successful_sync_at = utcnow()
    return created, {key: result.logo_candidates for key, result in results.items()}, candidates


def _ingest_directory(
    settings, registry, indexer, run_id, counts, owner, created, blocked, candidates=None
):
    settings.job_pdf_path.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        p for p in settings.job_pdf_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    counts["discovered"] = len(paths)
    with Session(registry.engine) as session:
        documents = list(session.scalars(select(JobDocument)))
        versions = list(session.scalars(select(DocumentVersion)))
    for input_path in paths:
        if input_path.name in blocked:
            continue
        registry.renew_lease(owner)
        try:
            path = resolve_pdf(settings, input_path)
            matches = [doc for doc in documents if Path(doc.source_path).resolve() == path]
            if not matches:
                pdf_hash = digest(path.read_bytes())
                matching_ids = {v.document_id for v in versions if v.pdf_sha256 == pdf_hash}
                renamed = [
                    doc
                    for doc in documents
                    if doc.document_id in matching_ids and not Path(doc.source_path).exists()
                ]
                if len(renamed) > 1:
                    raise ValueError(
                        "Ambiguous PDF rename; register it with an explicit document ID"
                    )
                matches = renamed
            if not matches and path.name == ORIGINAL_PDF_NAME:
                registry.save_document(DEFAULT_JOB_ID, "assignment-job-description", path)
                with Session(registry.engine) as session:
                    matches = [session.get(JobDocument, "assignment-job-description")]
            if not matches:
                if path.stat().st_size > 25_000_000:
                    raise ValueError("Document exceeds the 25 MB ingestion limit")
                metadata = PdfReader(path).metadata
                job_id = "local-" + digest(path.name)[:24]
                registry.source({"provider": "local", "board": "pdfs", "company": "Local PDFs"})
                registry.save_job(
                    {
                        "job_id": job_id,
                        "source_id": "local:pdfs",
                        "external_job_id": path.name,
                        "canonical_url": "",
                        "company_name": (metadata.author if metadata else None) or "Local PDF",
                        "title": (metadata.title if metadata else None) or path.stem,
                        "description_text": "\n".join(text for _, text in pdf_pages(path)),
                        "locations": [],
                    }
                )
                created.add(job_id)
                document_id = "pdf-" + digest(job_id)[:32]
                registry.save_document(job_id, document_id, path)
                with Session(registry.engine) as session:
                    matches = [session.get(JobDocument, document_id)]
            for doc in matches:
                job = registry.get_job(doc.job_id)
                if job["status"] in ("excluded", "closed"):
                    continue
                if job["status"] == "removed":
                    with Session(registry.engine) as session, session.begin():
                        session.get(Job, doc.job_id).status = "active"
                result = indexer.index(
                    job_id=doc.job_id,
                    document_id=doc.document_id,
                    pdf_path=path,
                    source_url=job["canonical_url"],
                    job_snapshot=(candidates or {}).get(doc.job_id),
                )
                action = (
                    "added"
                    if doc.job_id in created
                    else "unchanged"
                    if result["status"] == "unchanged"
                    else "updated"
                )
                counts[action] += 1
                registry.record_item(
                    run_id, action, job_id=doc.job_id, version_id=result["version_id"]
                )
        except Exception as exc:
            counts["failed"] += 1
            registry.record_item(run_id, "ingest_pdf", error=exc)
            log_event(
                "pdf_ingestion_failed",
                level=logging.WARNING,
                component="app.ingestion",
                run_id=run_id,
                error_code=type(exc).__name__,
            )


def sync_jobs(settings, *, offline=False, trigger="manual", cleanup=False, client=None):
    settings.require_api_key()
    configure_logging(settings.log_level)
    registry = JobRegistry(settings)
    counts = Counter(
        discovered=0, added=0, updated=0, unchanged=0, closed=0, failed=0, removed_documents=0
    )
    try:
        with (
            registry.lease() as owner,
            TraceRecorder(registry.engine).trace(
                None,
                str(uuid4()),
                component="ingestion",
                operation="sync_jobs",
                attributes={"trigger": trigger},
            ) as trace,
        ):
            previous = (
                json.loads(settings.job_manifest_path.read_text())
                if settings.job_manifest_path.exists()
                else {"jobs": []}
            )
            run_id = registry.start_run(trigger)
            trace.attributes["ingestion_run_id"] = run_id
            missing = set()
            try:
                created, missing = _restore_metadata(
                    settings, registry, previous, run_id, counts, require_files=offline
                )
                indexer = DocumentIndexer(settings, registry, lease_owner=owner)
                candidates = {}
                if not offline:
                    owned = client is None
                    client = client or httpx.Client(
                        timeout=settings.request_timeout,
                        headers={
                            "User-Agent": "RecruitmentCourseJobCorpus/1.0 (read-only public jobs)"
                        },
                        follow_redirects=False,
                    )
                    try:
                        collected, logo_hints, candidates = _collect(
                            settings, registry, run_id, counts, owner, client
                        )
                        created |= collected
                        registry.renew_lease(owner)
                        logo_counts = refresh_company_logos(
                            settings,
                            registry,
                            client,
                            sources=load_sources(settings.job_sources_path),
                            hints=logo_hints,
                        )
                        counts.update(logo_counts)
                    finally:
                        if owned:
                            client.close()
                blocked = {
                    Path(record["pdf_path"]).name
                    for record in previous["jobs"]
                    if record["document_id"] in missing
                }
                _ingest_directory(
                    settings, registry, indexer, run_id, counts, owner, created, blocked, candidates
                )
                counts["removed_documents"] = len(registry.disable_missing_documents())
                with Session(registry.engine) as session, session.begin():
                    for job in session.scalars(select(Job).where(Job.status == "active")):
                        if not session.scalar(
                            select(JobDocument.document_id)
                            .where(JobDocument.job_id == job.job_id, JobDocument.enabled.is_(True))
                            .limit(1)
                        ):
                            job.status = "removed"
                if cleanup:
                    counts["removed_vectors"] = indexer.cleanup()
            except Exception as exc:
                counts["failed"] += 1
                registry.record_item(run_id, "run", error=exc)
                log_event(
                    "ingestion_failed",
                    level=logging.WARNING,
                    component="app.ingestion",
                    error_code=type(exc).__name__,
                )
            manifest = manifest_paths(registry.manifest(settings))
            if counts["failed"]:
                present = {r["document_id"] for r in manifest["jobs"]}
                for record in previous["jobs"]:
                    if record["document_id"] not in present and (
                        record["document_id"] in missing or (ROOT / record["pdf_path"]).exists()
                    ):
                        manifest["jobs"].append({**record, "rebuild_status": "pending"})
            registry.finish_run(run_id, dict(counts))
            atomic_json(settings.job_manifest_path, manifest)
            report = {
                "run_id": run_id,
                "embedding_model": settings.embedding_model,
                "collection": settings.index_collection,
                "status": "partial" if counts["failed"] else "complete",
                "counts": dict(counts),
                "catalog_jobs": len(registry.list_jobs()),
                "manifest": str(settings.job_manifest_path),
            }
            atomic_json(settings.ingestion_report_path / f"{run_id}.json", report)
            trace.attributes["counts"] = dict(counts)
            if counts["failed"] or counts["logos_failed"]:
                trace.fail("IngestionPartial")
            return report
    finally:
        registry.close()
