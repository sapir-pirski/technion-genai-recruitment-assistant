"""Interview availability and durable booking receipts."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.config import DEFAULT_JOB_ID
from app.storage.database import Base


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
