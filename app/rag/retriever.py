from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.guardrails import reject
from app.observability.tracing import EmbeddingClient
from app.rag.qualifications import (
    LABELS,
    qualification_groups,
    qualification_items,
    select_qualifications,
)
from app.rag.store import PostgresVectorStore


def vectorstore(settings: Settings, *, engine=None) -> PostgresVectorStore:
    embedding = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.require_api_key(),
        request_timeout=settings.request_timeout,
        max_retries=settings.max_retries,
        **({"dimensions": settings.embedding_dimensions} if settings.embedding_dimensions else {}),
    )

    if getattr(embedding, "client", None) is not None:
        embedding.client = EmbeddingClient(embedding.client, settings.embedding_model)
    return PostgresVectorStore(settings, embedding, engine=engine)


class JobRetriever:
    def __init__(self, settings: Settings):
        from app.ingestion.registry import JobRegistry

        self.settings = settings
        self.registry = JobRegistry(settings)
        try:
            self.store = vectorstore(settings, engine=self.registry.engine)
        except Exception:
            self.registry.close()
            raise

    def search(self, query: str, job_id=None) -> list[dict]:
        from app.ingestion.models import DEFAULT_JOB_ID

        job_id = job_id or DEFAULT_JOB_ID
        versions = self.registry.active_versions(job_id, self.settings.index_collection)
        if not versions:
            return []
        # Include more sections to distinguish required from preferred
        # qualifications while keeping short job descriptions within the model budget.
        docs = self.store.similarity_search(query, k=6, job_id=job_id, version_ids=versions)
        excerpts = [
            {
                "id": doc.metadata["chunk_id"],
                "job_id": doc.metadata["job_id"],
                "page": doc.metadata["page"],
                "source": doc.metadata["source"],
                "source_url": doc.metadata.get("source_url", ""),
                "text": doc.page_content,
            }
            for doc in docs
        ]
        groups = qualification_groups(query)
        complete = self.store.get(job_id=job_id, version_ids=versions, limit=257)
        if len(complete["ids"]) > 256:
            if not groups:
                return excerpts
            reject(
                "qualification_coverage",
                "retrieval",
                "document_budget",
                "The full qualification section needs recruiter review.",
            )
        ordered = [
            {**metadata, "id": key, "text": text}
            for key, text, metadata in zip(
                complete["ids"], complete["documents"], complete["metadatas"], strict=True
            )
        ]
        items = select_qualifications(query, qualification_items(ordered, set(LABELS)))
        if items:
            needed = {source for item in items for source in item["source_ids"]}
            present = {row["id"] for row in excerpts}
            excerpts.extend(row for row in ordered if row["id"] in needed - present)
            excerpts[0]["qualification_items"] = items
        return excerpts

    def close(self):
        self.registry.close()

    def references(self, source_ids, job_id):
        """Resolve saved citations without a model call or another vector search."""
        if not source_ids:
            return []
        records = self.store.get(ids=source_ids, job_id=job_id)
        by_id = dict(zip(records["ids"], records["metadatas"], strict=True))
        return [{"id": key, **by_id[key]} for key in dict.fromkeys(source_ids) if key in by_id]
