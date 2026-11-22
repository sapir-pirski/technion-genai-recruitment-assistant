import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.ingestion.description import FORMAT_VERSION
from app.ingestion.registry import JobRegistry
from app.rag.models import RagChunk


def startup_index_ready(settings, registry):
    """Compare authoritative PDF inventory, saved manifest and published chunks."""
    try:
        records = registry.manifest(settings)["jobs"]
        saved = json.loads(settings.job_manifest_path.read_text())["jobs"]
        files = {
            path.name: path
            for path in settings.job_pdf_path.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        }
        expected = {Path(row["pdf_path"]).name: row for row in records}
        if not files or set(files) != set(expected) or len(expected) != len(records):
            return False
        saved_hashes = {Path(row["pdf_path"]).name: row["pdf_sha256"] for row in saved}
        for name, path in files.items():
            row = expected[name]
            if path.resolve().parent != settings.job_pdf_path.resolve():
                return False
            if Path(row["source_path"]).resolve() != path.resolve():
                return False
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != row["pdf_sha256"] or saved_hashes.get(name) != digest:
                return False
            if row["status"] == "active" and (
                row["collection"] != settings.index_collection
                or row["render_version"] != FORMAT_VERSION
            ):
                return False
        return set(saved_hashes) == set(files) and index_status(settings, registry)["complete"]
    except (OSError, ValueError, KeyError, TypeError):
        return False


def operate(settings, command, *, if_needed=False):
    registry = JobRegistry(settings)
    try:
        if command == "list-jobs":
            return registry.list_jobs(active_only=False)
        if command == "ingestion-status":
            return registry.recent_runs()
        if command == "bootstrap":
            from app.ingestion.sync import sync_jobs

            if if_needed and startup_index_ready(settings, registry):
                return {"status": "ready", "ingestion_skipped": True, "counts": {"failed": 0}}
            return sync_jobs(settings, offline=True, cleanup=True)
        return index_status(settings, registry)
    finally:
        registry.close()


def index_status(settings, registry):
    manifest = registry.manifest(settings)
    checks = []
    for item in manifest["jobs"]:
        if item["status"] != "active":
            continue
        with registry.engine.connect() as connection:
            actual = (
                list(
                    connection.scalars(
                        select(RagChunk.chunk_id).where(
                            RagChunk.collection_id == settings.index_collection,
                            RagChunk.document_version_id == item["document_version_id"],
                        )
                    )
                )
                if item["document_version_id"]
                else []
            )
        expected = {f"{item['document_version_id']}:{index}" for index in range(item["chunks"])}
        checks.append(
            {
                "job_id": item["job_id"],
                "document_id": item["document_id"],
                "expected_chunks": item["chunks"],
                "actual_chunks": len(actual),
                "complete": bool(item["document_version_id"]) and set(actual) == expected,
            }
        )
    missing = [
        job["job_id"]
        for job in registry.list_jobs()
        if not registry.active_versions(job["job_id"], settings.index_collection)
    ]
    return {
        "missing_job_indexes": missing,
        "collection": settings.index_collection,
        "documents": len(checks),
        "complete": bool(checks) and not missing and all(check["complete"] for check in checks),
        "checks": checks,
    }
