from langchain_openai import OpenAIEmbeddings

from app.config import Settings
from app.observability.tracing import EmbeddingClient
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
        self.store = vectorstore(settings, engine=self.registry.engine)

    def search(self, query: str, job_id=None) -> list[dict]:
        from app.ingestion.models import DEFAULT_JOB_ID

        job_id = job_id or DEFAULT_JOB_ID
        versions = self.registry.active_versions(job_id, self.settings.index_collection)
        if not versions:
            return []
        docs = self.store.similarity_search(query, k=3, job_id=job_id, version_ids=versions)
        return [
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

    def close(self):
        self.registry.close()

    def references(self, source_ids, job_id):
        """Resolve saved citations without a model call or another vector search."""
        if not source_ids:
            return []
        records = self.store.get(ids=source_ids, job_id=job_id)
        by_id = dict(zip(records["ids"], records["metadatas"], strict=True))
        return [{"id": key, **by_id[key]} for key in dict.fromkeys(source_ids) if key in by_id]
