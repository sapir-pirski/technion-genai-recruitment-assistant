from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("The application requires PostgreSQL with pgvector")
    return create_engine(
        parsed.set(drivername="postgresql+psycopg"), pool_pre_ping=True, hide_parameters=True
    )


def initialize_database(engine):
    """Create the final schema on a new database; existing data is never replaced."""
    from app.ingestion.models import CatalogBase
    from app.rag.models import RagChunk, RagCollection  # noqa: F401
    from app.storage.models import Base

    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(73489123)"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
        CatalogBase.metadata.create_all(connection)
        Base.metadata.create_all(connection)
