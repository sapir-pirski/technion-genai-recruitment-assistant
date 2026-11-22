"""Bounded recovery for initialization and read-only workspace loading."""

import logging
import time
from dataclasses import dataclass

import psycopg
from psycopg_pool import PoolTimeout
from sqlalchemy.exc import DBAPIError

from app.ingestion.models import DEFAULT_JOB_ID
from app.observability.logging import log_event


def transient_database_error(error):
    original = error.orig if isinstance(error, DBAPIError) else error
    code = getattr(original, "sqlstate", None)
    if code:
        return code.startswith("08") or code in {
            "53300",
            "57P01",
            "57P02",
            "57P03",
            "57P05",
        }
    if isinstance(original, PoolTimeout):
        return True
    if isinstance(error, DBAPIError) and error.connection_invalidated:
        return True
    if isinstance(original, (psycopg.OperationalError, psycopg.InterfaceError)):
        # libpq connection failures may not have a server SQLSTATE. Do not treat
        # authentication, TLS configuration, or arbitrary operational errors as transient.
        message = str(original).lower()
        return any(
            fragment in message
            for fragment in (
                "connection refused",
                "timeout expired",
                "connection is closed",
                "connection reset by peer",
                "server closed the connection",
                "ssl connection has been closed",
                "network is unreachable",
                "temporary failure in name resolution",
            )
        )
    return False


class StartupFailure(Exception):
    def __init__(self, stage, retryable):
        super().__init__("Workspace startup failed")
        self.stage = stage
        self.retryable = retryable


@dataclass
class Workspace:
    service: object
    state: dict
    messages: list
    choices: dict
    job: dict | None = None
    logo: dict | None = None
    downloads: list | None = None


def load_workspace(settings, factory, conversation_id, selected_job_id, downloads, on_retry):
    """Retry complete read snapshots; never submit or replay a candidate request."""
    for attempt in range(1, 4):
        stage = "service"
        try:
            service = factory()
            stage = "checkpoint"
            state = service.state(conversation_id)
            stage = "messages"
            messages = service.conversations.messages(conversation_id)
            stage = "jobs"
            jobs = sorted(
                service.registry.list_jobs(collection=settings.index_collection),
                key=lambda job: (
                    job["job_id"] != DEFAULT_JOB_ID,
                    job["company_name"],
                    job["title"],
                ),
            )
            choices = {job["job_id"]: job for job in jobs}
            workspace = Workspace(service, state, messages, choices)
            selected = state.get("job_id") or selected_job_id
            if not state.get("job_id") and selected not in choices:
                selected = next(iter(choices), None)
            if selected not in choices:
                return workspace
            stage = "job"
            workspace.job = service.registry.get_job(selected, collection=settings.index_collection)
            if not workspace.job:
                return workspace
            stage = "logo"
            workspace.logo = service.registry.company_logo(selected)
            stage = "documents"
            workspace.downloads = downloads(service, settings, workspace.job)
            if attempt > 1:
                log_event("startup_recovered", component="app.streamlit", attempt=attempt)
            return workspace
        except Exception as error:
            retryable = transient_database_error(error)
            retry = retryable and attempt < 3
            log_event(
                "startup_retry" if retry else "service_initialization_failed",
                level=logging.WARNING if retry else logging.ERROR,
                component="app.streamlit",
                stage=stage,
                attempt=attempt,
                error_code=type(error).__name__,
                retryable=retryable,
            )
            if not retry:
                raise StartupFailure(stage, retryable) from error
            on_retry(attempt + 1)
            time.sleep(0.5 * attempt)
