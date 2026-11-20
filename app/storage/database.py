from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.schema import CreateSchema

DEFAULT_SCHEMA = "recruitment"


class Base(DeclarativeBase):
    """Shared metadata for the seven core application tables."""

    pass


def make_engine(url: str, *, schema=DEFAULT_SCHEMA) -> Engine:
    if not url:
        raise ValueError("Set DATABASE_URL before running the database-backed application")
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise ValueError("The application requires PostgreSQL with pgvector")
    return create_engine(
        parsed.set(drivername="postgresql+psycopg"),
        pool_pre_ping=True,
        hide_parameters=True,
        execution_options={"schema_translate_map": {None: schema}},
    )


def initialize_database(engine):
    """Create only core tables in an isolated schema; never alter legacy public tables."""
    from app.rag import models as rag_models  # noqa: F401
    from app.scheduling import models as scheduling_models  # noqa: F401
    from app.storage import models as conversation_models  # noqa: F401

    schema = engine.get_execution_options()["schema_translate_map"][None]
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(73489123)"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
        connection.execute(CreateSchema(schema, if_not_exists=True))
        Base.metadata.create_all(connection)
