from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
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

from app.ingestion.models import DEFAULT_JOB_ID, CatalogBase


class Base(DeclarativeBase):
    pass


class Slot(Base):
    __tablename__ = "interview_slots"
    __table_args__ = (
        UniqueConstraint("starts_at", "position", "job_id", name="uq_slot_job_position_time"),
        CheckConstraint(
            "duration_minutes IS NULL OR duration_minutes > 0", name="ck_slot_duration_positive"
        ),
    )
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # UTC-naive timestamps at the persistence boundary; API timestamps include UTC.
    starts_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    job_id: Mapped[str] = mapped_column(String(80), default=DEFAULT_JOB_ID, index=True)
    position: Mapped[str] = mapped_column(String(40), default="Python Dev")
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    duration_minutes: Mapped[int | None] = mapped_column(Integer)


class Booking(Base):
    __tablename__ = "interview_bookings"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(80), default=DEFAULT_JOB_ID, index=True)
    slot_id: Mapped[int] = mapped_column(ForeignKey("interview_slots.id"), unique=True)
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True)
    request_id: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Conversation(CatalogBase):
    __tablename__ = "conversations"
    __table_args__ = (CheckConstraint("status IN ('active', 'closed')"),)
    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.job_id"), index=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    timezone: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationMessage(CatalogBase):
    __tablename__ = "conversation_messages"
    __table_args__ = (
        UniqueConstraint("conversation_id", "sequence_no"),
        UniqueConstraint("conversation_id", "request_id", "role"),
        CheckConstraint("role IN ('user', 'assistant')"),
        CheckConstraint("sequence_no > 0"),
        CheckConstraint(
            "predicted_action IS NULL OR predicted_action IN ('continue', 'schedule', 'end')"
        ),
    )
    message_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), index=True
    )
    sequence_no: Mapped[int] = mapped_column(Integer)
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    predicted_action: Mapped[str | None] = mapped_column(String(16))
    # Durable response receipt: retries can return the booking/result without another model call.
    response_data: Mapped[dict | None] = mapped_column(JSONB)


class CandidateSource(CatalogBase):
    __tablename__ = "candidate_sources"
    __table_args__ = (
        CheckConstraint("source_kind IN ('introduction', 'linkedin_reference', 'file')"),
        CheckConstraint("status IN ('pending', 'extracted', 'ready', 'reference', 'failed')"),
    )
    source_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), index=True
    )
    request_id: Mapped[str] = mapped_column(String(64))
    source_kind: Mapped[str] = mapped_column(String(24))
    filename: Mapped[str | None] = mapped_column(String(200))
    media_type: Mapped[str | None] = mapped_column(String(100))
    source_url: Mapped[str | None] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(String(64))
    pipeline_version: Mapped[str] = mapped_column(String(100))
    byte_count: Mapped[int] = mapped_column(Integer)
    input_data: Mapped[bytes | None] = mapped_column(LargeBinary)
    extracted_text: Mapped[str | None] = mapped_column(Text)
    facts: Mapped[dict | None] = mapped_column(JSONB)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    retryable: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    error_code: Mapped[str | None] = mapped_column(String(100))
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CandidateProfile(CatalogBase):
    __tablename__ = "candidate_profiles"
    __table_args__ = (
        Index(
            "uq_candidate_profile_current",
            "conversation_id",
            unique=True,
            postgresql_where=text("is_current"),
        ),
    )
    profile_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.conversation_id"), index=True
    )
    request_id: Mapped[str] = mapped_column(String(64))
    input_fingerprint: Mapped[str] = mapped_column(String(64))
    source_ids: Mapped[list] = mapped_column(JSONB)
    data: Mapped[dict] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(String(20))
    is_current: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TraceSpan(CatalogBase):
    __tablename__ = "trace_spans"
    __table_args__ = (
        CheckConstraint("status IN ('running', 'complete', 'failed')"),
        CheckConstraint("duration_ms >= 0"),
        CheckConstraint("input_tokens IS NULL OR input_tokens >= 0"),
        CheckConstraint("output_tokens IS NULL OR output_tokens >= 0"),
        CheckConstraint("cached_input_tokens IS NULL OR cached_input_tokens >= 0"),
        CheckConstraint(
            "cached_input_tokens IS NULL OR input_tokens IS NULL OR cached_input_tokens <= input_tokens"
        ),
    )
    span_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(36), index=True)
    parent_span_id: Mapped[str | None] = mapped_column(
        ForeignKey("trace_spans.span_id", deferrable=True, initially="DEFERRED")
    )
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.conversation_id"), index=True
    )
    request_id: Mapped[str] = mapped_column(String(64), index=True)
    component: Mapped[str] = mapped_column(String(80))
    operation: Mapped[str] = mapped_column(String(80))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16))
    model: Mapped[str | None] = mapped_column(String(200))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cached_input_tokens: Mapped[int | None] = mapped_column(Integer)
    error_code: Mapped[str | None] = mapped_column(String(100))
    attributes: Mapped[dict] = mapped_column(JSONB, default=dict)


class DatasetConversation(CatalogBase):
    __tablename__ = "dataset_conversations"
    __table_args__ = (
        CheckConstraint("split IN ('train', 'validation', 'test')"),
        CheckConstraint("conversation_id > 0"),
        CheckConstraint("source_sha256 ~ '^[0-9a-f]{64}$'"),
    )
    dataset_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    split: Mapped[str] = mapped_column(String(16), index=True)
    source_sha256: Mapped[str] = mapped_column(String(64))
    # Preserve supplied contact metadata; it is excluded from model inputs and live exports.
    source_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)


class DatasetTurn(CatalogBase):
    __tablename__ = "dataset_turns"
    __table_args__ = (
        ForeignKeyConstraint(
            ["dataset_version", "conversation_id"],
            ["dataset_conversations.dataset_version", "dataset_conversations.conversation_id"],
        ),
        CheckConstraint("turn_id > 0"),
        CheckConstraint("speaker IN ('candidate', 'recruiter')"),
        CheckConstraint(
            "(speaker = 'candidate' AND expected_action IS NULL) OR (speaker = 'recruiter' AND expected_action IS NOT NULL AND expected_action IN ('continue', 'schedule', 'end'))"
        ),
    )
    dataset_version: Mapped[str] = mapped_column(String(100), primary_key=True)
    conversation_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    turn_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    speaker: Mapped[str] = mapped_column(String(16))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    text: Mapped[str] = mapped_column(Text)
    expected_action: Mapped[str | None] = mapped_column(String(16))
