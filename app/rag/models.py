from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.ingestion.models import CatalogBase


class RagCollection(CatalogBase):
    __tablename__ = "rag_collections"
    collection_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    embedding_model: Mapped[str] = mapped_column(String(160))
    dimensions: Mapped[int] = mapped_column(Integer)


class RagChunk(CatalogBase):
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
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.job_id"), index=True)
    document_id: Mapped[str | None] = mapped_column(ForeignKey("job_documents.document_id"))
    document_version_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_versions.document_version_id"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector())
    dimensions: Mapped[int] = mapped_column(Integer)
    attributes: Mapped[dict] = mapped_column("metadata", JSONB)
