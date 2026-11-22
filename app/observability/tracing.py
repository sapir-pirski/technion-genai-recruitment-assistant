import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from uuid import uuid4

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.observability.logging import correlation, log_event
from app.observability.pricing import estimate_cost
from app.storage.models import TraceSpan

_context = ContextVar("execution_trace", default=None)


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


@dataclass
class Span:
    component: str
    operation: str
    model: str | None = None
    span_id: str = field(default_factory=lambda: str(uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_ms: float = 0
    status: str = "running"
    error_code: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    attributes: dict = field(default_factory=dict)

    def fail(self, error_code):
        self.status, self.error_code = "failed", error_code

    def usage(self, message, *, embedding=False):
        if isinstance(message, dict):
            usage = message.get("usage") or {}
            metadata = {}
        else:
            metadata = getattr(message, "response_metadata", {}) or {}
            # Prefer provider fields: normalized adapters can replace missing counts with zero.
            raw_usage = metadata.get("token_usage")
            usage = (
                raw_usage
                if isinstance(raw_usage, dict)
                else getattr(message, "usage_metadata", None) or {}
            )
            response_id = getattr(message, "id", None)
            if isinstance(response_id, str):
                self.attributes["response_id"] = response_id
            returned_model = metadata.get("model_name")
            if isinstance(returned_model, str):
                self.attributes["requested_model"] = self.model
                self.model = returned_model
        self.input_tokens = _count(usage.get("input_tokens", usage.get("prompt_tokens")))
        self.output_tokens = (
            None
            if embedding
            else _count(usage.get("output_tokens", usage.get("completion_tokens")))
        )
        details = (
            usage.get("input_token_details")
            or usage.get("input_tokens_details")
            or usage.get("prompt_tokens_details")
            or {}
        )
        self.cached_input_tokens = _count(details.get("cache_read", details.get("cached_tokens")))
        if (
            self.input_tokens is not None
            and self.cached_input_tokens is not None
            and self.cached_input_tokens > self.input_tokens
        ):
            self.cached_input_tokens = None
        self.attributes["usage_reported"] = self.input_tokens is not None
        output_details = (
            usage.get("output_token_details")
            or usage.get("output_tokens_details")
            or usage.get("completion_tokens_details")
            or {}
        )
        # Reasoning is a subset of output_tokens, never a second billable total.
        self.attributes["reasoning_tokens"] = _count(
            output_details.get("reasoning", output_details.get("reasoning_tokens"))
        )
        self.attributes.update(
            estimate_cost(
                self.model,
                self.input_tokens,
                self.output_tokens,
                self.cached_input_tokens,
                embedding=embedding,
            )
        )


@contextmanager
def span(component, operation, *, model=None, attributes=None):
    current = _context.get()
    item = Span(component, operation, model=model, attributes=dict(attributes or {}))
    started = time.perf_counter()
    token = log_token = None
    if current is not None:
        trace_id, conversation_id, request_id, parent_id, rows, recorder = current
        rows.append((item, parent_id))
        token = _context.set((trace_id, conversation_id, request_id, item.span_id, rows, recorder))
        log_token = correlation.set(
            {
                "trace_id": trace_id,
                "span_id": item.span_id,
                "conversation_id": conversation_id,
                "request_id": request_id,
            }
        )
        recorder.save(trace_id, conversation_id, request_id, [(item, parent_id)])
    try:
        yield item
    except Exception as error:
        item.fail(type(error).__name__)
        raise
    finally:
        item.finished_at = datetime.now(UTC)
        item.duration_ms = max(0, (time.perf_counter() - started) * 1000)
        if item.status == "running":
            item.status = "complete"
        if current is not None:
            recorder.save(trace_id, conversation_id, request_id, [(item, parent_id)])
        log_event(
            "span_failed" if item.status == "failed" else "span_completed",
            level=logging.WARNING if item.status == "failed" else logging.DEBUG,
            component=f"app.{component}",
            duration_ms=round(item.duration_ms, 3),
            error_code=item.error_code,
        )
        if token is not None:
            _context.reset(token)
            correlation.reset(log_token)


class TraceRecorder:
    def __init__(self, engine):
        self.engine = engine

    @contextmanager
    def trace(
        self,
        conversation_id,
        request_id,
        *,
        component="conversation",
        operation="turn",
        attributes=None,
    ):
        trace_id = str(uuid4())
        rows = []
        token = _context.set((trace_id, conversation_id, request_id, None, rows, self))
        try:
            with span(component, operation, attributes=attributes) as root:
                log_event("trace_started", component="app.observability")
                yield root
        finally:
            try:
                if self.save(trace_id, conversation_id, request_id, rows):
                    log_event(
                        "trace_saved",
                        component="app.observability",
                        trace_id=trace_id,
                        conversation_id=conversation_id,
                        request_id=request_id,
                    )
            finally:
                _context.reset(token)

    def save(self, trace_id, conversation_id, request_id, rows):
        try:
            values = [
                dict(
                    trace_id=trace_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    parent_span_id=parent_id,
                    **{
                        key: getattr(item, key)
                        for key in (
                            "span_id",
                            "component",
                            "operation",
                            "started_at",
                            "finished_at",
                            "duration_ms",
                            "status",
                            "model",
                            "input_tokens",
                            "output_tokens",
                            "cached_input_tokens",
                            "error_code",
                            "attributes",
                        )
                    },
                )
                for item, parent_id in rows
            ]
            statement = insert(TraceSpan).values(values)
            statement = statement.on_conflict_do_update(
                index_elements=["span_id"],
                set_={
                    column.name: statement.excluded[column.name]
                    for column in TraceSpan.__table__.columns
                    if column.name != "span_id"
                },
            )
            with Session(self.engine) as session, session.begin():
                session.execute(statement)
            return True
        except Exception as error:
            # Diagnostics must not cause another model call or undo a booking.
            log_event(
                "trace_persistence_failed",
                level=logging.ERROR,
                component="app.observability",
                trace_id=trace_id,
                conversation_id=conversation_id,
                request_id=request_id,
                error_code=type(error).__name__,
            )
            return False


def traced(component, operation):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with span(
                component,
                operation,
                attributes={
                    key: kwargs[key]
                    for key in ("job_id", "document_id")
                    if isinstance(kwargs.get(key), str)
                },
            ) as item:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get("status"), str):
                    item.attributes["outcome"] = result["status"]
                return result

        return wrapped

    return decorate


class EmbeddingClient:
    """Observe existing synchronous requests without creating extra API calls."""

    def __init__(self, client, model):
        self.client, self.model = client, model

    def create(self, *args, **kwargs):
        with span("embedding", "create", model=self.model) as item:
            response = self.client.create(*args, **kwargs)
            item.usage(
                response if isinstance(response, dict) else response.model_dump(), embedding=True
            )
            return response
