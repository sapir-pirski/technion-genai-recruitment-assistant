from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ingestion.models import (
    ASSIGNMENT_DOCUMENT_ID,
    DEFAULT_JOB_ID,
    DocumentVersion,
    Job,
    JobDocument,
    JobSource,
    SyncLease,
)
from app.ingestion.normalize import digest
from app.storage.database import initialize_database, make_engine


def utcnow():
    return datetime.now(UTC).replace(tzinfo=None)


class JobRegistry:
    def __init__(self, settings):
        self.engine = make_engine(settings.database_url)
        self.collection = settings.index_collection
        initialize_database(self.engine)
        self.ensure_original()

    def close(self):
        self.engine.dispose()

    def ensure_original(self):
        with Session(self.engine) as session, session.begin():
            if session.get(Job, DEFAULT_JOB_ID):
                return
            session.add(
                JobSource(
                    source_id="local:assignment",
                    provider="local",
                    board_identifier="assignment",
                    company_name="Tech company",
                    enabled=True,
                )
            )
            session.flush()
            session.add(
                Job(
                    job_id=DEFAULT_JOB_ID,
                    source_id="local:assignment",
                    external_job_id="original",
                    canonical_url="",
                    company_name="Tech company",
                    title="Python Developer",
                    description_text="",
                    first_seen_at=utcnow(),
                    last_seen_at=utcnow(),
                    status="active",
                    scheduling_enabled=True,
                )
            )

    @staticmethod
    def _location_texts(job):
        return [
            location["location_text"]
            for location in sorted(
                job.locations,
                key=lambda location: (location["location_text"], location["location_id"]),
            )
        ]

    def get_job(self, job_id, *, collection=None):
        if job_id != DEFAULT_JOB_ID:
            return None
        with Session(self.engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return None
            result = {column.name: getattr(job, column.name) for column in Job.__table__.columns}
            result["locations"] = self._location_texts(job)
            if collection or self.collection:
                published = session.execute(
                    select(DocumentVersion.presentation)
                    .join(
                        JobDocument,
                        JobDocument.document_id == DocumentVersion.document_id,
                    )
                    .where(
                        JobDocument.job_id == job_id,
                        JobDocument.document_id == ASSIGNMENT_DOCUMENT_ID,
                        JobDocument.enabled.is_(True),
                        DocumentVersion.collection_name == (collection or self.collection),
                        DocumentVersion.is_active.is_(True),
                        DocumentVersion.status == "complete",
                    )
                    .order_by(JobDocument.document_id)
                    .limit(1)
                ).first()
                presentation = published[0] if published else None
                if (
                    collection
                    and published is None
                    and session.scalar(
                        select(JobDocument.document_id)
                        .where(
                            JobDocument.job_id == job_id,
                            JobDocument.document_id == ASSIGNMENT_DOCUMENT_ID,
                            JobDocument.enabled.is_(True),
                        )
                        .limit(1)
                    )
                ):
                    return None
                if collection:
                    self._present(result, presentation)
                if presentation:
                    result["render_version"] = presentation.get("format_version")
                    result["description_notices"] = presentation.get("notices", [])
            return result

    @staticmethod
    def _present(job, presentation):
        if presentation:
            job.update(presentation.get("job", {}))
            job["description_text"] = presentation["description_text"]
            job["description_format"] = presentation.get("description_format", "markdown")
            job["description_notices"] = presentation.get("notices", [])

    def list_jobs(self, *, active_only=True, collection=None):
        job = self.get_job(DEFAULT_JOB_ID, collection=collection)
        if job is None or (active_only and job["status"] != "active"):
            return []
        fields = (
            "job_id",
            "title",
            "company_name",
            "status",
            "canonical_url",
            "scheduling_enabled",
            "locations",
            "workplace_type",
        )
        return [{key: job[key] for key in fields}]

    def documents(self, job_id, *, collection=None):
        """List the selected job's published PDF inputs for the candidate view."""
        if job_id != DEFAULT_JOB_ID:
            return []
        with Session(self.engine) as session:
            if collection:
                published = [
                    {
                        "document_id": doc.document_id,
                        "path": version.pdf_path,
                        "url": doc.source_url,
                        "data": version.pdf_data,
                        "sha256": version.pdf_sha256,
                    }
                    for doc, version in session.execute(
                        select(JobDocument, DocumentVersion)
                        .join(
                            DocumentVersion,
                            DocumentVersion.document_id == JobDocument.document_id,
                        )
                        .where(
                            JobDocument.job_id == job_id,
                            JobDocument.document_id == ASSIGNMENT_DOCUMENT_ID,
                            JobDocument.enabled.is_(True),
                            DocumentVersion.collection_name == collection,
                            DocumentVersion.is_active.is_(True),
                            DocumentVersion.status == "complete",
                        )
                        .order_by(JobDocument.document_id)
                    )
                ]
                if published:
                    return published
            return [
                {
                    "document_id": doc.document_id,
                    "path": doc.source_path,
                    "url": doc.source_url,
                }
                for doc in session.scalars(
                    select(JobDocument)
                    .where(
                        JobDocument.job_id == job_id,
                        JobDocument.document_id == ASSIGNMENT_DOCUMENT_ID,
                        JobDocument.enabled.is_(True),
                    )
                    .order_by(JobDocument.document_id)
                )
            ]

    def save_document(self, job_id, document_id, path, url=""):
        if job_id != DEFAULT_JOB_ID:
            raise ValueError("Only the assignment Python Developer role is supported")
        with Session(self.engine) as session, session.begin():
            doc = session.get(JobDocument, document_id)
            if doc and doc.job_id != job_id:
                raise ValueError("Document identity cannot move between jobs")
            if not doc:
                doc = JobDocument(
                    document_id=document_id, job_id=job_id, source_path=str(path), source_url=url
                )
                session.add(doc)
            doc.enabled, doc.source_path, doc.source_url = True, str(path), url

    def active_versions(self, job_id, collection):
        if job_id != DEFAULT_JOB_ID:
            return []
        with Session(self.engine) as session:
            return list(
                session.scalars(
                    select(DocumentVersion.document_version_id)
                    .join(JobDocument, DocumentVersion.document_id == JobDocument.document_id)
                    .join(Job, JobDocument.job_id == Job.job_id)
                    .where(
                        JobDocument.job_id == job_id,
                        JobDocument.document_id == ASSIGNMENT_DOCUMENT_ID,
                        JobDocument.enabled.is_(True),
                        Job.status == "active",
                        DocumentVersion.collection_name == collection,
                        DocumentVersion.is_active.is_(True),
                        DocumentVersion.status == "complete",
                    )
                )
            )

    def current_version(self, document_id, collection):
        with Session(self.engine) as session:
            return session.scalar(
                select(DocumentVersion.document_version_id).where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.collection_name == collection,
                    DocumentVersion.is_active.is_(True),
                )
            )

    def current_presentation(self, document_id, collection, pdf_sha256):
        with Session(self.engine) as session:
            return session.scalar(
                select(DocumentVersion.presentation).where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.collection_name == collection,
                    DocumentVersion.is_active.is_(True),
                    DocumentVersion.pdf_sha256 == pdf_sha256,
                    DocumentVersion.status == "complete",
                )
            )

    def stage(self, values):
        with Session(self.engine) as session, session.begin():
            session.get(JobDocument, values["document_id"], with_for_update=True)
            version = session.get(DocumentVersion, values["document_version_id"])
            if not version:
                version = DocumentVersion(**values)
                session.add(version)
            else:
                # A repair of the current version must be unpublished until complete.
                version.is_active = False
                for name, value in values.items():
                    setattr(version, name, value)

    def refresh_version_file(self, version_id, pdf_path, *, presentation=None):
        """Refresh an unchanged index's file and presentation without re-embedding."""
        with Session(self.engine) as session, session.begin():
            version = session.get(DocumentVersion, version_id)
            version.pdf_path = str(pdf_path)
            version.pdf_sha256 = digest(Path(pdf_path).read_bytes())
            version.pdf_data = Path(pdf_path).read_bytes()
            if presentation is not None:
                version.presentation = presentation
                job = session.get(Job, session.get(JobDocument, version.document_id).job_id)
                job.description_text = presentation["description_text"]
                job.description_format = presentation.get("description_format", "markdown")

    def publish(self, document_id, collection, version_id, count, dimensions, *, lease_owner=None):
        with Session(self.engine) as session, session.begin():
            if lease_owner is not None:
                changed = session.execute(
                    update(SyncLease)
                    .where(
                        SyncLease.lease_id == "jobs",
                        SyncLease.owner == lease_owner,
                        SyncLease.expires_at > utcnow(),
                    )
                    .values(expires_at=utcnow() + timedelta(hours=2))
                )
                if changed.rowcount != 1:
                    raise ValueError("Ingestion lease lost before publication")
            document = session.get(JobDocument, document_id, with_for_update=True)
            version = session.get(DocumentVersion, version_id)
            if (
                document is None
                or version is None
                or version.document_id != document_id
                or version.collection_name != collection
            ):
                raise ValueError("Document version does not match the publication target")
            if count != version.expected_chunk_count:
                raise ValueError("Incomplete document version")
            # Lock the parent and retire its previous version before activation.
            # Readers see either publication through the transaction's atomic commit.
            session.execute(
                update(DocumentVersion)
                .where(
                    DocumentVersion.document_id == document_id,
                    DocumentVersion.collection_name == collection,
                    DocumentVersion.is_active.is_(True),
                )
                .values(is_active=False)
            )
            version.status, version.indexed_chunk_count = "complete", count
            version.indexed_at, version.embedding_dimensions = utcnow(), dimensions
            version.error_code = None
            version.is_active = True
            if version.presentation:
                job = session.get(Job, session.get(JobDocument, document_id).job_id)
                job.description_text = version.presentation["description_text"]
                job.description_format = version.presentation.get("description_format", "markdown")

    def fail_version(self, version_id, error):
        with Session(self.engine) as session, session.begin():
            version = session.get(DocumentVersion, version_id)
            if version and not version.is_active:
                version.status, version.error_code = "failed", type(error).__name__

    @contextmanager
    def lease(self):
        owner = str(uuid4())
        now = utcnow()
        with Session(self.engine) as session, session.begin():
            existing = session.get(SyncLease, "jobs")
            if existing is None:
                try:
                    with session.begin_nested():
                        session.add(
                            SyncLease(
                                lease_id="jobs", owner=owner, expires_at=now + timedelta(hours=2)
                            )
                        )
                        session.flush()
                except IntegrityError as exc:
                    raise ValueError("Another ingestion run is active") from exc
            else:
                changed = session.execute(
                    update(SyncLease)
                    .where(SyncLease.lease_id == "jobs", SyncLease.expires_at < now)
                    .values(owner=owner, expires_at=now + timedelta(hours=2))
                )
                if changed.rowcount != 1:
                    raise ValueError("Another ingestion run is active")
        try:
            yield owner
        finally:
            with Session(self.engine) as session, session.begin():
                session.execute(
                    delete(SyncLease).where(SyncLease.lease_id == "jobs", SyncLease.owner == owner)
                )

    def renew_lease(self, owner):
        with Session(self.engine) as session, session.begin():
            changed = session.execute(
                update(SyncLease)
                .where(
                    SyncLease.lease_id == "jobs",
                    SyncLease.owner == owner,
                    SyncLease.expires_at > utcnow(),
                )
                .values(expires_at=utcnow() + timedelta(hours=2))
            )
            if changed.rowcount != 1:
                raise ValueError("Ingestion lease expired")

    def manifest(self, settings):
        """Describe only the assignment PDF, including when legacy jobs remain in the DB."""
        with Session(self.engine) as session:
            job = session.get(Job, DEFAULT_JOB_ID)
            doc = session.get(JobDocument, ASSIGNMENT_DOCUMENT_ID)
            records = []
            if doc and doc.enabled and doc.job_id == DEFAULT_JOB_ID:
                version = session.scalar(
                    select(DocumentVersion).where(
                        DocumentVersion.document_id == ASSIGNMENT_DOCUMENT_ID,
                        DocumentVersion.collection_name == settings.index_collection,
                        DocumentVersion.is_active.is_(True),
                    )
                )
                pdf_path = Path(version.pdf_path if version else doc.source_path)
                presentation = version.presentation if version else None
                records.append(
                    {
                        "job_id": DEFAULT_JOB_ID,
                        "document_id": ASSIGNMENT_DOCUMENT_ID,
                        "document_version_id": version.document_version_id if version else None,
                        "title": job.title,
                        "status": job.status,
                        "source_path": str(doc.source_path),
                        "pdf_path": str(pdf_path),
                        "pdf_sha256": version.pdf_sha256 if version else None,
                        "description_text": presentation["description_text"]
                        if presentation
                        else "",
                        "description_format": "markdown",
                        "render_version": presentation.get("format_version")
                        if presentation
                        else None,
                        "chunks": version.indexed_chunk_count if version else 0,
                        "collection": settings.index_collection,
                        "index_status": version.status if version else "pending",
                    }
                )
            return {
                "schema_version": 2,
                "generated_at": utcnow().isoformat() + "Z",
                "embedding_model": settings.embedding_model,
                "jobs": records,
            }
