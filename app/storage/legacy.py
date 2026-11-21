"""Explicit, non-destructive copy from a legacy schema into an empty core schema."""

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import connect
from psycopg.rows import dict_row
from sqlalchemy import MetaData, Table, func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.config import DEFAULT_JOB_ID
from app.storage.database import initialize_database, make_engine

CORE_TABLES = ("interview_slots", "conversations", "interview_bookings", "conversation_messages")
CHECKPOINT_TABLES = ("checkpoint_blobs", "checkpoints", "checkpoint_writes")


def copy_legacy(settings, *, source_schema="public", dry_run=False):
    if source_schema == settings.database_schema:
        raise ValueError("Source and target schemas must differ")
    engine = make_engine(settings.database_url, schema=settings.database_schema)
    try:
        initialize_database(engine)
        with connect(
            settings.postgres_dsn, autocommit=True, prepare_threshold=0, row_factory=dict_row
        ) as connection:
            PostgresSaver(connection).setup()
        inspector = inspect(engine)
        available = set(inspector.get_table_names(schema=source_schema))
        names = [*CORE_TABLES, *CHECKPOINT_TABLES]
        if not set(names) <= available:
            raise ValueError(
                "Source schema must contain legacy conversation/calendar and checkpoint tables"
            )
        sources = {
            name: Table(
                name, MetaData(), schema=source_schema, autoload_with=engine, resolve_fks=False
            )
            for name in names
        }
        targets = {
            name: Table(
                name,
                MetaData(),
                schema=settings.database_schema,
                autoload_with=engine,
                resolve_fks=False,
            )
            for name in names
        }
        counts = {}
        with engine.connect().execution_options(isolation_level="REPEATABLE READ") as connection:
            transaction = connection.begin()
            try:
                # Block target writers until the emptiness check and complete copy commit together.
                quote = engine.dialect.identifier_preparer.format_table
                for name in sorted(names):
                    table = targets[name]
                    connection.execute(
                        text(f"LOCK TABLE {quote(table)} IN SHARE ROW EXCLUSIVE MODE")
                    )
                    if connection.scalar(select(func.count()).select_from(table)):
                        raise ValueError(
                            "Target calendar, conversations and checkpoints must be empty"
                        )
                conversation_ids = select(sources["conversations"].c.conversation_id).where(
                    sources["conversations"].c.job_id == DEFAULT_JOB_ID
                )
                for name in names:
                    source, target = sources[name], targets[name]
                    columns = [column.name for column in target.columns]
                    if not set(columns) <= set(source.columns.keys()):
                        raise ValueError(
                            f"Legacy {name} columns differ; migrate with the matching LangGraph version"
                        )
                    query = select(*(source.c[column] for column in columns))
                    if name in CHECKPOINT_TABLES:
                        query = query.where(source.c.thread_id.in_(conversation_ids))
                    elif name == "conversation_messages":
                        query = query.where(source.c.conversation_id.in_(conversation_ids))
                    else:
                        query = query.where(source.c.job_id == DEFAULT_JOB_ID)
                    expected = connection.scalar(select(func.count()).select_from(query.subquery()))
                    connection.execute(target.insert().from_select(columns, query))
                    actual = connection.scalar(select(func.count()).select_from(target))
                    if actual != expected:
                        raise ValueError(f"Copy count mismatch for {name}")
                    counts[name] = actual
                if dry_run:
                    transaction.rollback()
                else:
                    connection.execute(
                        text(
                            "SELECT setval(pg_get_serial_sequence(:table, 'id'), "
                            "COALESCE((SELECT max(id) FROM "
                            + quote(targets["interview_slots"])
                            + "), 1), "
                            "EXISTS (SELECT 1 FROM " + quote(targets["interview_slots"]) + "))"
                        ),
                        {"table": f"{settings.database_schema}.interview_slots"},
                    )
                    transaction.commit()
            except IntegrityError as error:
                transaction.rollback()
                raise ValueError(
                    "Legacy data violates target constraints; no rows were copied"
                ) from error
            except Exception:
                transaction.rollback()
                raise
        return {
            "source_schema": source_schema,
            "target_schema": settings.database_schema,
            "dry_run": dry_run,
            "copied_rows": counts,
        }
    finally:
        engine.dispose()
