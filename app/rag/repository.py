from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.config import ASSIGNMENT_DOCUMENT_ID, DEFAULT_JOB_ID
from app.rag.description import digest
from app.rag.models import JobDescription
from app.storage.database import initialize_database, make_engine


def utcnow():
    return datetime.now(UTC)


class JobDescriptionRepository:
    def __init__(self, settings):
        self.engine = make_engine(settings.database_url, schema=settings.database_schema)
        self.collection = settings.index_collection
        initialize_database(self.engine)

    def close(self):
        self.engine.dispose()

    def get_job(self, job_id, *, collection=None):
        if job_id != DEFAULT_JOB_ID:
            return None
        result = {
            "job_id": DEFAULT_JOB_ID,
            "title": "Python Developer",
            "company_name": "Tech company",
            "canonical_url": "",
            "description_text": "",
            "description_format": "markdown",
            "status": "active",
            "scheduling_enabled": True,
            "locations": [],
            "workplace_type": None,
        }
        with Session(self.engine) as session:
            document = session.get(JobDescription, collection or self.collection)
            if document:
                presentation = document.presentation
                result.update(presentation.get("job", {}))
                result.update(
                    description_text=presentation["description_text"],
                    description_format=presentation.get("description_format", "markdown"),
                    render_version=presentation.get("format_version"),
                    description_notices=presentation.get("notices", []),
                )
        return result

    def list_jobs(self, *, active_only=True, collection=None):
        job = self.get_job(DEFAULT_JOB_ID, collection=collection)
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
        if job_id != DEFAULT_JOB_ID:
            return []
        with Session(self.engine) as session:
            document = session.get(JobDescription, collection or self.collection)
            return (
                [
                    {
                        "document_id": ASSIGNMENT_DOCUMENT_ID,
                        "path": document.pdf_path,
                        "url": document.source_url,
                        "data": document.pdf_data,
                        "sha256": document.pdf_sha256,
                    }
                ]
                if document
                else []
            )

    def active_versions(self, job_id, collection):
        if job_id != DEFAULT_JOB_ID:
            return []
        from app.rag.models import RagChunk

        with Session(self.engine) as session:
            document = session.get(JobDescription, collection)
            if document is None:
                return []
            ids = set(
                session.scalars(
                    select(RagChunk.chunk_id).where(
                        RagChunk.collection_id == collection,
                        RagChunk.job_id == job_id,
                        RagChunk.document_version_id == document.document_version_id,
                    )
                )
            )
            expected = {f"{document.document_version_id}:{i}" for i in range(document.chunk_count)}
            return [document.document_version_id] if ids == expected else []

    def current_version(self, document_id, collection):
        with Session(self.engine) as session:
            document = session.get(JobDescription, collection)
            return document.document_version_id if document else None

    def current_presentation(self, document_id, collection, pdf_sha256):
        with Session(self.engine) as session:
            document = session.get(JobDescription, collection)
            return document.presentation if document and document.pdf_sha256 == pdf_sha256 else None

    def refresh_version_file(self, version_id, pdf_path, *, presentation=None):
        with Session(self.engine) as session, session.begin():
            document = session.get(JobDescription, self.collection)
            if document is None or document.document_version_id != version_id:
                raise ValueError("Published document version changed")
            document.pdf_path = str(pdf_path)
            document.pdf_data = Path(pdf_path).read_bytes()
            document.pdf_sha256 = digest(document.pdf_data)
            if presentation is not None:
                document.presentation = presentation

    def publish(self, values):
        from app.rag.models import RagChunk

        collection, version = values["collection_name"], values["document_version_id"]
        with Session(self.engine) as session, session.begin():
            rows = session.execute(
                select(RagChunk.chunk_id, RagChunk.dimensions).where(
                    RagChunk.collection_id == collection,
                    RagChunk.document_id == ASSIGNMENT_DOCUMENT_ID,
                    RagChunk.job_id == DEFAULT_JOB_ID,
                    RagChunk.document_version_id == version,
                )
            ).all()
            expected = {f"{version}:{i}" for i in range(values["chunk_count"])}
            if {row.chunk_id for row in rows} != expected or any(
                row.dimensions != values["embedding_dimensions"] for row in rows
            ):
                raise ValueError("Incomplete document index")
            statement = insert(JobDescription).values(**values, indexed_at=utcnow())
            session.execute(
                statement.on_conflict_do_update(
                    index_elements=[JobDescription.collection_name],
                    set_={
                        key: value
                        for key, value in statement.excluded.items()
                        if key != "collection_name"
                    },
                )
            )

    @contextmanager
    def index_lock(self):
        # A connection-scoped PostgreSQL lock needs no lease table or heartbeat.
        key = {
            "key": f"{self.engine.get_execution_options()['schema_translate_map'][None]}:{self.collection}"
        }
        with self.engine.connect() as connection:
            acquired = connection.scalar(
                text("SELECT pg_try_advisory_lock(hashtextextended(:key, 731000))"), key
            )
            connection.commit()
            if not acquired:
                raise ValueError("Another ingestion run is active")
            try:
                yield
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:key, 731000))"), key
                )
                connection.commit()

    def manifest(self, settings):
        with Session(self.engine) as session:
            document = session.get(JobDescription, settings.index_collection)
            records = (
                [
                    {
                        "job_id": DEFAULT_JOB_ID,
                        "document_id": ASSIGNMENT_DOCUMENT_ID,
                        "document_version_id": document.document_version_id,
                        "title": "Python Developer",
                        "status": "active",
                        "source_path": document.pdf_path,
                        "pdf_path": document.pdf_path,
                        "pdf_sha256": document.pdf_sha256,
                        "description_text": document.presentation["description_text"],
                        "description_format": document.presentation.get(
                            "description_format", "markdown"
                        ),
                        "render_version": document.presentation.get("format_version"),
                        "chunks": document.chunk_count,
                        "collection": settings.index_collection,
                        "index_status": "complete",
                    }
                ]
                if document
                else []
            )
        return {
            "schema_version": 3,
            "generated_at": utcnow().isoformat(),
            "embedding_model": settings.embedding_model,
            "jobs": records,
        }
