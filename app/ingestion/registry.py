from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, literal, select, update
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ingestion.models import (
    DEFAULT_JOB_ID,
    Company,
    DocumentVersion,
    IngestionRun,
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
                    company_name="Assignment",
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
                    company_name="Assignment",
                    title="Python Developer",
                    description_text="",
                    first_seen_at=utcnow(),
                    last_seen_at=utcnow(),
                    status="active",
                    scheduling_enabled=True,
                )
            )

    def source(self, source):
        from app.ingestion.logos import company_identity

        key = f"{source['provider']}:{source['board']}"
        identity, website = company_identity(source)
        with Session(self.engine) as session, session.begin():
            company_id = "company-" + digest(identity)[:24] if identity else None
            if company_id:
                company = session.get(Company, company_id)
                if not company:
                    company = Company(
                        company_id=company_id,
                        identity_key=identity,
                        name=source["company"],
                        website_url=website,
                    )
                    session.add(company)
                    session.flush()
                else:
                    company.name = source["company"]
                    if website:
                        company.website_url = website
            record = session.get(JobSource, key)
            if not record:
                record = JobSource(
                    source_id=key,
                    provider=source["provider"],
                    board_identifier=source["board"],
                    company_name=source["company"],
                    enabled=True,
                )
                session.add(record)
                session.flush()
            record.enabled = source.get("enabled", True)
            record.company_name = source["company"]
            record.company_id = company_id
            session.execute(update(Job).where(Job.source_id == key).values(company_id=company_id))
            return {"source_id": key, "company_id": company_id}

    def save_job(self, value):
        fields = (
            "source_id",
            "external_job_id",
            "canonical_url",
            "company_name",
            "title",
            "description_text",
            "description_format",
            "employment_type",
            "seniority",
            "workplace_type",
            "compensation",
            "source_published_at",
            "source_updated_at",
        )
        with Session(self.engine) as session, session.begin():
            job = session.get(Job, value["job_id"])
            created = job is None
            if created:
                job = Job(
                    job_id=value["job_id"],
                    first_seen_at=utcnow(),
                    last_seen_at=utcnow(),
                    scheduling_enabled=False,
                )
                session.add(job)
            for name in fields:
                setattr(
                    job, name, value.get(name, "plain" if name == "description_format" else None)
                )
            source = session.get(JobSource, value["source_id"])
            job.company_id = source.company_id if source else None
            job.status, job.closed_at, job.missing_scans = "active", None, 0
            job.last_seen_at = utcnow()
            job.locations = [
                {
                    "location_id": digest(f"{job.job_id}:{index}"),
                    **{
                        key: location.get(key)
                        for key in ("location_text", "city", "region", "country_code")
                    },
                }
                for index, location in enumerate(value.get("locations", []))
            ]
            if any(not location["location_text"] for location in job.locations):
                raise ValueError("Each job location needs location_text")
            return created

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
                        .where(JobDocument.job_id == job_id, JobDocument.enabled.is_(True))
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
        with Session(self.engine) as session:
            query = select(Job).order_by(Job.company_name, Job.title, Job.job_id)
            if active_only:
                query = query.where(Job.status == "active")
            jobs = list(session.scalars(query))
            result = [
                {
                    "job_id": job.job_id,
                    "title": job.title,
                    "company_name": job.company_name,
                    "status": job.status,
                    "canonical_url": job.canonical_url,
                    "scheduling_enabled": job.scheduling_enabled,
                    "locations": self._location_texts(job),
                    "workplace_type": job.workplace_type,
                }
                for job in jobs
            ]
            if collection:
                presentations = {}
                for job_id, presentation in session.execute(
                    select(JobDocument.job_id, DocumentVersion.presentation)
                    .join(
                        DocumentVersion,
                        DocumentVersion.document_id == JobDocument.document_id,
                    )
                    .where(
                        JobDocument.enabled.is_(True),
                        DocumentVersion.collection_name == collection,
                        DocumentVersion.is_active.is_(True),
                        DocumentVersion.status == "complete",
                    )
                    .order_by(JobDocument.document_id)
                ):
                    presentations.setdefault(job_id, presentation)
                for job in result:
                    self._present(job, presentations.get(job["job_id"]))
                registered = set(
                    session.scalars(select(JobDocument.job_id).where(JobDocument.enabled.is_(True)))
                )
                result = [
                    job
                    for job in result
                    if job["job_id"] not in registered or job["job_id"] in presentations
                ]
            return result

    def documents(self, job_id, *, collection=None):
        """List the selected job's published PDF inputs for the candidate view."""
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
                    .where(JobDocument.job_id == job_id, JobDocument.enabled.is_(True))
                    .order_by(JobDocument.document_id)
                )
            ]

    def company_logo(self, job_id):
        """Read just the selected company's cached image; never fetch from the UI."""
        with Session(self.engine) as session:
            company = session.scalar(
                select(Company)
                .join(Job, Job.company_id == Company.company_id)
                .where(Job.job_id == job_id)
            )
            if not company or not company.logo_data:
                return None
            return {
                "data": company.logo_data,
                "mime_type": company.logo_mime_type,
                "sha256": company.logo_sha256,
                "company_id": company.company_id,
            }

    def save_document(self, job_id, document_id, path, url=""):
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
        with Session(self.engine) as session:
            return list(
                session.scalars(
                    select(DocumentVersion.document_version_id)
                    .join(JobDocument, DocumentVersion.document_id == JobDocument.document_id)
                    .join(Job, JobDocument.job_id == Job.job_id)
                    .where(
                        JobDocument.job_id == job_id,
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

    def reconcile_source(self, source_id, all_ids):
        """Only call after a complete successful source scan. Two misses confirm closure."""
        closed = []
        with Session(self.engine) as session, session.begin():
            for job in session.scalars(select(Job).where(Job.source_id == source_id)):
                if job.external_job_id in all_ids:
                    job.last_seen_at, job.missing_scans = utcnow(), 0
                elif job.status == "active":
                    job.missing_scans += 1
                    if job.missing_scans >= 2:
                        job.status, job.closed_at = "closed", utcnow()
                        closed.append(job.job_id)
        return closed

    def disable_missing_documents(self):
        from pathlib import Path

        removed = []
        with Session(self.engine) as session, session.begin():
            for doc in session.scalars(select(JobDocument).where(JobDocument.enabled.is_(True))):
                if not Path(doc.source_path).exists():
                    doc.enabled = False
                    removed.append(doc.document_id)
        return removed

    def start_run(self, trigger):
        run_id = str(uuid4())
        with Session(self.engine) as session, session.begin():
            session.add(
                IngestionRun(
                    run_id=run_id, trigger=trigger, started_at=utcnow(), status="running", counts={}
                )
            )
        return run_id

    def record_item(self, run_id, action, *, job_id=None, version_id=None, error=None):
        item = {
            "item_id": str(uuid4()),
            "job_id": job_id,
            "document_version_id": version_id,
            "action": action,
            "status": "failed" if error else "complete",
            "attempt_count": 1,
            "error_code": type(error).__name__ if error else None,
        }
        with Session(self.engine) as session, session.begin():
            changed = session.execute(
                update(IngestionRun)
                .where(IngestionRun.run_id == run_id)
                .values(items=IngestionRun.items + literal([item], type_=JSONB))
            )
            if changed.rowcount != 1:
                raise ValueError("Unknown ingestion run")

    def finish_run(self, run_id, counts):
        with Session(self.engine) as session, session.begin():
            run = session.get(IngestionRun, run_id)
            run.finished_at, run.counts = utcnow(), counts
            run.status = (
                "partial" if counts.get("failed") or counts.get("logos_failed") else "complete"
            )

    def recent_runs(self, limit=10):
        with Session(self.engine) as session:
            return [
                {
                    "run_id": run.run_id,
                    "status": run.status,
                    "started_at": run.started_at.isoformat() + "Z",
                    "counts": run.counts,
                }
                for run in session.scalars(
                    select(IngestionRun).order_by(IngestionRun.started_at.desc()).limit(limit)
                )
            ]

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
        with Session(self.engine) as session:
            records = []
            for job in session.scalars(select(Job).order_by(Job.job_id)):
                company = session.get(Company, job.company_id) if job.company_id else None
                docs = list(
                    session.scalars(
                        select(JobDocument).where(
                            JobDocument.job_id == job.job_id, JobDocument.enabled.is_(True)
                        )
                    )
                )
                for doc in docs:
                    version = session.scalar(
                        select(DocumentVersion).where(
                            DocumentVersion.document_id == doc.document_id,
                            DocumentVersion.collection_name == settings.index_collection,
                            DocumentVersion.is_active.is_(True),
                        )
                    )
                    pdf_path = Path(version.pdf_path if version else doc.source_path)
                    record = {
                        "job_id": job.job_id,
                        "document_id": doc.document_id,
                        "document_version_id": version.document_version_id if version else None,
                        "company_name": job.company_name,
                        "company_id": job.company_id,
                        "company_website_url": company.website_url if company else None,
                        "title": job.title,
                        "description_text": job.description_text,
                        "description_format": job.description_format,
                        "render_version": version.presentation.get("format_version")
                        if version and version.presentation
                        else None,
                        "locations": [
                            {
                                key: location.get(key)
                                for key in ("location_text", "city", "region", "country_code")
                            }
                            for location in job.locations
                        ],
                        "employment_type": job.employment_type,
                        "workplace_type": job.workplace_type,
                        "seniority": job.seniority,
                        "compensation": job.compensation,
                        "source_published_at": job.source_published_at,
                        "canonical_url": job.canonical_url,
                        "source_id": job.source_id,
                        "external_job_id": job.external_job_id,
                        "status": job.status,
                        "last_seen_at": job.last_seen_at.isoformat() + "Z",
                        "indexed_at": version.indexed_at.isoformat() + "Z"
                        if version and version.indexed_at
                        else None,
                        "source_updated_at": job.source_updated_at,
                        "pdf_sha256": version.pdf_sha256
                        if version
                        else digest(pdf_path.read_bytes())
                        if pdf_path.is_file()
                        else None,
                        "source_path": str(doc.source_path),
                        "pdf_path": str(pdf_path),
                        "text_sha256": version.text_sha256 if version else None,
                        "chunks": version.indexed_chunk_count if version else 0,
                        "collection": settings.index_collection,
                        "index_status": version.status if version else "pending",
                    }
                    if version and version.presentation:
                        presentation = version.presentation
                        metadata = presentation.get("job", {})
                        record.update(
                            {key: value for key, value in metadata.items() if key != "locations"}
                        )
                        record["locations"] = [
                            {
                                "location_text": location,
                                "city": None,
                                "region": None,
                                "country_code": None,
                            }
                            for location in metadata.get("locations", [])
                        ]
                        record["description_text"] = presentation["description_text"]
                        record["description_format"] = presentation.get(
                            "description_format", "markdown"
                        )
                        record["description_notices"] = presentation.get("notices", [])
                    records.append(record)
            return {
                "schema_version": 2,
                "generated_at": utcnow().isoformat() + "Z",
                "embedding_model": settings.embedding_model,
                "jobs": records,
            }
