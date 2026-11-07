from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_JOB_ID = "assignment-python-developer"


class CatalogBase(DeclarativeBase):
    pass


class Company(CatalogBase):
    __tablename__ = "companies"
    company_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    identity_key: Mapped[str] = mapped_column(String(300), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    website_url: Mapped[str | None] = mapped_column(Text)
    logo_source_url: Mapped[str | None] = mapped_column(Text)
    logo_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    logo_mime_type: Mapped[str | None] = mapped_column(String(80))
    logo_sha256: Mapped[str | None] = mapped_column(String(64))
    logo_pipeline_version: Mapped[str | None] = mapped_column(String(40))
    logo_etag: Mapped[str | None] = mapped_column(Text)
    logo_last_modified: Mapped[str | None] = mapped_column(Text)
    logo_checked_at: Mapped[datetime | None] = mapped_column(DateTime)
    logo_updated_at: Mapped[datetime | None] = mapped_column(DateTime)
    logo_status: Mapped[str] = mapped_column(String(30), default="missing")
    logo_error_code: Mapped[str | None] = mapped_column(String(100))


class JobSource(CatalogBase):
    __tablename__ = "job_sources"
    source_id: Mapped[str] = mapped_column(String(120), primary_key=True)
    provider: Mapped[str] = mapped_column(String(30))
    board_identifier: Mapped[str] = mapped_column(String(120))
    company_name: Mapped[str] = mapped_column(String(200))
    company_id: Mapped[str | None] = mapped_column(ForeignKey("companies.company_id"), index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_successful_sync_at: Mapped[datetime | None] = mapped_column(DateTime)


class Job(CatalogBase):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("source_id", "external_job_id"),
        CheckConstraint("jsonb_typeof(locations) = 'array'", name="jobs_locations_array"),
    )
    job_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("job_sources.source_id"))
    external_job_id: Mapped[str] = mapped_column(String(200))
    canonical_url: Mapped[str] = mapped_column(Text)
    company_name: Mapped[str] = mapped_column(String(200))
    company_id: Mapped[str | None] = mapped_column(ForeignKey("companies.company_id"), index=True)
    title: Mapped[str] = mapped_column(Text)
    description_text: Mapped[str] = mapped_column(Text)
    description_format: Mapped[str] = mapped_column(String(20), default="plain")
    employment_type: Mapped[str | None] = mapped_column(String(100))
    seniority: Mapped[str | None] = mapped_column(String(100))
    workplace_type: Mapped[str | None] = mapped_column(String(50))
    compensation: Mapped[dict | None] = mapped_column(JSON)
    locations: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))
    status: Mapped[str] = mapped_column(String(20), default="active")
    source_published_at: Mapped[str | None] = mapped_column(String(80))
    source_updated_at: Mapped[str | None] = mapped_column(String(80))
    first_seen_at: Mapped[datetime] = mapped_column(DateTime)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime)
    missing_scans: Mapped[int] = mapped_column(Integer, default=0)
    scheduling_enabled: Mapped[bool] = mapped_column(Boolean, default=False)


class JobDocument(CatalogBase):
    __tablename__ = "job_documents"
    document_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), index=True)
    source_path: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class DocumentVersion(CatalogBase):
    __tablename__ = "document_versions"
    __table_args__ = (
        Index(
            "document_versions_one_active",
            "document_id",
            "collection_name",
            unique=True,
            postgresql_where=text("is_active"),
        ),
        CheckConstraint(
            "NOT is_active OR (status = 'complete' AND "
            "indexed_chunk_count = expected_chunk_count AND indexed_at IS NOT NULL)",
            name="document_versions_active_complete",
        ),
    )
    document_version_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("job_documents.document_id"), index=True)
    pdf_path: Mapped[str] = mapped_column(Text)
    pdf_sha256: Mapped[str] = mapped_column(String(64))
    text_sha256: Mapped[str] = mapped_column(String(64))
    pipeline_version: Mapped[str] = mapped_column(String(80))
    embedding_model: Mapped[str] = mapped_column(String(150))
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer)
    collection_name: Mapped[str] = mapped_column(String(150))
    expected_chunk_count: Mapped[int] = mapped_column(Integer)
    indexed_chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime)
    error_code: Mapped[str | None] = mapped_column(String(100))
    presentation: Mapped[dict | None] = mapped_column(JSONB)
    pdf_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))


class IngestionRun(CatalogBase):
    __tablename__ = "ingestion_runs"
    __table_args__ = (
        CheckConstraint("jsonb_typeof(items) = 'array'", name="ingestion_runs_items_array"),
    )
    run_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trigger: Mapped[str] = mapped_column(String(30))
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20))
    counts: Mapped[dict] = mapped_column(JSON, default=dict)
    items: Mapped[list] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))


class SyncLease(CatalogBase):
    __tablename__ = "sync_leases"
    lease_id: Mapped[str] = mapped_column(String(30), primary_key=True)
    owner: Mapped[str] = mapped_column(String(36))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
