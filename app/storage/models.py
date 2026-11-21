"""Durable conversation history and response receipts."""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.config import DEFAULT_JOB_ID
from app.storage.database import Base


class Conversation(Base):
    __tablename__ = "conversations"
    __table_args__ = (CheckConstraint("status IN ('active', 'closed')"),)
    conversation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(80), default=DEFAULT_JOB_ID, index=True)
    status: Mapped[str] = mapped_column(String(16), default="active")
    timezone: Mapped[str] = mapped_column(String(100))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ConversationMessage(Base):
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
