import json
import logging
import sys
import threading
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import wraps

correlation: ContextVar[dict | None] = ContextVar("log_correlation", default=None)
_lock = threading.Lock()
_fields = {
    "rule_id",
    "stage",
    "outcome",
    "reason_code",
    "decision",
    "consultation_count",
    "stop_reason",
    "trace_id",
    "span_id",
    "conversation_id",
    "request_id",
    "error_code",
    "duration_ms",
    "job_id",
    "source_id",
    "run_id",
    "suggestion_id",
    "answer_request_id",
    "suggestion_count",
    "suggestion_source",
    "validation_outcome",
    "slot_id",
    "slot_count",
    "timezone",
    "search_kind",
    "calendar_action",
}


class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Render explicit fields only. Never serialize exception text, arbitrary extras or args.
        data = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "event": getattr(record, "event", "application_event"),
            "component": record.name,
        }
        for name in _fields:
            value = record.__dict__.get(name, (correlation.get() or {}).get(name))
            if isinstance(value, (str, int, float)):
                data[name] = value
        return json.dumps(data, ensure_ascii=False)


class JsonStderrHandler(logging.Handler):
    def emit(self, record):
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except (OSError, ValueError):
            pass


def configure_logging(level="INFO"):
    with _lock:
        logger = logging.getLogger("app")
        if not any(getattr(handler, "recruitment_json", False) for handler in logger.handlers):
            handler = JsonStderrHandler()
            handler.recruitment_json = True
            handler.setFormatter(JsonFormatter())
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False


def log_event(event, *, level=logging.INFO, component="app", **fields):
    logging.getLogger(component).log(
        level,
        event,
        extra={"event": event, **{key: value for key, value in fields.items() if key in _fields}},
    )


def log_errors(component):
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as error:
                log_event(
                    f"{fn.__name__}_failed",
                    level=logging.ERROR,
                    component=component,
                    error_code=type(error).__name__,
                    conversation_id=kwargs.get(
                        "conversation_id", args[1] if len(args) > 1 else None
                    ),
                    request_id=kwargs.get("request_id"),
                )
                raise

        return wrapped

    return decorate
