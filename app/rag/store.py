"""RAG chunks and exact cosine retrieval in the application's PostgreSQL database."""

from langchain_core.documents import Document
from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.rag.models import RagChunk, RagCollection
from app.storage.database import initialize_database, make_engine


class PostgresVectorStore:
    def __init__(self, settings, embeddings, *, engine=None):
        self.settings, self.embeddings = settings, embeddings
        self.engine = engine if engine is not None else make_engine(settings.database_url)
        self._owns_engine = engine is None
        initialize_database(self.engine)
        self.collection_id = settings.index_collection

    def _query(self, *, ids=None, version_id=None, job_id=None):
        query = select(RagChunk).where(RagChunk.collection_id == self.collection_id)
        if ids is not None:
            query = query.where(RagChunk.chunk_id.in_(ids))
        if version_id is not None:
            query = query.where(RagChunk.document_version_id == version_id)
        if job_id is not None:
            query = query.where(RagChunk.job_id == job_id)
        return query

    def get(self, *, ids=None, version_id=None, job_id=None, include=None):
        with Session(self.engine) as session:
            rows = session.scalars(
                self._query(ids=ids, version_id=version_id, job_id=job_id).order_by(
                    RagChunk.chunk_id
                )
            ).all()
            result = {
                "ids": [row.chunk_id for row in rows],
                "documents": [row.text for row in rows],
                "metadatas": [row.attributes for row in rows],
            }
            if include and "embeddings" in include:
                result["embeddings"] = [list(row.embedding) for row in rows]
            return result

    def add_documents(self, documents, ids):
        vectors = self.embeddings.embed_documents([doc.page_content for doc in documents])
        if not vectors or len(vectors) != len(documents) or len(ids) != len(documents):
            raise ValueError("Embedding response does not match the document batch")
        dimensions = len(vectors[0])
        if dimensions == 0 or any(len(vector) != dimensions for vector in vectors):
            raise ValueError("Embedding dimensions differ within the document batch")
        with Session(self.engine) as session, session.begin():
            session.execute(
                insert(RagCollection)
                .values(
                    collection_id=self.collection_id,
                    embedding_model=self.settings.embedding_identity,
                    dimensions=dimensions,
                )
                .on_conflict_do_nothing()
            )
            collection = session.scalar(
                select(RagCollection)
                .where(RagCollection.collection_id == self.collection_id)
                .with_for_update()
            )
            if (
                collection.dimensions != dimensions
                or collection.embedding_model != self.settings.embedding_identity
            ):
                raise ValueError("Embedding configuration does not match the existing collection")
            for key, doc, vector in zip(ids, documents, vectors, strict=True):
                metadata = doc.metadata
                values = {
                    "collection_id": self.collection_id,
                    "chunk_id": key,
                    "text": doc.page_content,
                    "embedding": vector,
                    "dimensions": dimensions,
                    "metadata": metadata,
                    "job_id": metadata.get("job_id"),
                    "document_id": metadata.get("document_id"),
                    "document_version_id": metadata.get("document_version_id"),
                }
                statement = insert(RagChunk.__table__).values(**values)
                session.execute(
                    statement.on_conflict_do_update(
                        index_elements=[RagChunk.collection_id, RagChunk.chunk_id],
                        set_={
                            key: value
                            for key, value in values.items()
                            if key not in ("collection_id", "chunk_id")
                        },
                    )
                )

    def update_metadata(self, *, ids, metadatas):
        with self.engine.begin() as connection:
            for key, metadata in zip(ids, metadatas, strict=True):
                connection.execute(
                    update(RagChunk)
                    .where(RagChunk.collection_id == self.collection_id, RagChunk.chunk_id == key)
                    .values(attributes=metadata)
                )

    def delete(self, *, ids=None, version_id=None):
        if ids is None and version_id is None:
            raise ValueError("Specify chunk IDs or a document version to delete")
        query = delete(RagChunk).where(RagChunk.collection_id == self.collection_id)
        if ids is not None:
            query = query.where(RagChunk.chunk_id.in_(ids))
        if version_id is not None:
            query = query.where(RagChunk.document_version_id == version_id)
        with self.engine.begin() as connection:
            connection.execute(query)

    def similarity_search(self, query, *, job_id, version_ids, k=3):
        vector = self.embeddings.embed_query(query)
        with Session(self.engine) as session:
            collection = session.get(RagCollection, self.collection_id)
            if collection is None:
                return []
            if collection.dimensions != len(vector):
                raise ValueError("Query embedding dimensions do not match the stored collection")
            rows = session.scalars(
                self._query(job_id=job_id)
                .where(RagChunk.document_version_id.in_(version_ids))
                .order_by(RagChunk.embedding.cosine_distance(vector), RagChunk.chunk_id)
                .limit(k)
            ).all()
            return [Document(page_content=row.text, metadata=row.attributes) for row in rows]

    def close(self):
        if self._owns_engine:
            self.engine.dispose()
