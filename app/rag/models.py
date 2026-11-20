from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.storage.database import Base


class RagCollection(Base):
    __tablename__ = "rag_collections"
    collection_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(160))
    dimensions: Mapped[int] = mapped_column(Integer)


class RagChunk(Base):
    __tablename__ = "rag_chunks"
    __table_args__ = (
        CheckConstraint(
            "vector_dims(embedding) = dimensions AND dimensions > 0", name="ck_embedding_dimensions"
        ),
    )
    collection_id: Mapped[str] = mapped_column(
        ForeignKey("rag_collections.collection_id"), primary_key=True
    )
    chunk_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    job_id: Mapped[str | None] = mapped_column(String(80), index=True)
    document_id: Mapped[str | None] = mapped_column(String(80))
    document_version_id: Mapped[str | None] = mapped_column(String(80), index=True)
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    dimensions: Mapped[int] = mapped_column(Integer)
    attributes: Mapped[dict] = mapped_column("metadata", JSONB)


class JobDescription(Base):
    __tablename__ = "job_descriptions"
    __table_args__ = (CheckConstraint("chunk_count > 0 AND embedding_dimensions > 0"),)
    collection_name: Mapped[str] = mapped_column(String(160), primary_key=True)
    document_version_id: Mapped[str] = mapped_column(String(80))
    pdf_path: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str] = mapped_column(Text, default="")
    pdf_sha256: Mapped[str] = mapped_column(String(64))
    pdf_data: Mapped[bytes] = mapped_column(LargeBinary)
    presentation: Mapped[dict] = mapped_column(JSONB)
    chunk_count: Mapped[int] = mapped_column(Integer)
    embedding_dimensions: Mapped[int] = mapped_column(Integer)
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
