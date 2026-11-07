"""Source ingestion subgraph. Nodes checkpoint references, never upload bytes."""

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from app.graph.state import ConversationState
from app.guardrails import redact_credentials
from app.observability.tracing import span, traced
from app.profiles import (
    ProfileReadError,
    bounded_text,
    grounded_sections,
    linkedin_reference,
    merge_profile,
    read_document,
    snapshot_fingerprint,
)


def should_ingest(state, repository):
    if repository is None:
        return False
    ids = state.get("profile_source_ids", [])
    return bool(ids or state.get("profile_version")) and (
        snapshot_fingerprint(ids) != state.get("profile_input_fingerprint")
        or not state.get("profile_version")
        or any(issue.get("retryable") for issue in state.get("profile_issues", []))
    )


def build_profile_ingestion(backend, repository):
    def rows(state):
        return repository.sources(state["conversation_id"], state.get("profile_source_ids", []))

    def failed(source, error):
        if source["source_kind"] == "linkedin_reference":
            repository.update_source(source["source_id"], source_url=None)
        repository.update_source(
            source["source_id"],
            status="failed",
            input_data=None,
            error_code=error.code if isinstance(error, ProfileReadError) else type(error).__name__,
            error_message=str(error)
            if isinstance(error, ProfileReadError)
            else "This source could not be processed. Please clarify its relevant details in chat or replace it.",
            retryable=isinstance(
                error,
                (
                    APIConnectionError,
                    APITimeoutError,
                    InternalServerError,
                    RateLimitError,
                    TimeoutError,
                    ConnectionError,
                ),
            )
            and source.get("attempt_count", 0) < 3,
        )

    @traced("candidate_profile", "validate_sources")
    def validate(state):
        get_stream_writer()(
            {"stage": "validate_sources", "label": "Checking your background information…"}
        )
        for source in rows(state):
            if source["status"] != "pending":
                continue
            with span(
                "candidate_profile",
                "validate_source",
                attributes={"source_id": source["source_id"], "source_kind": source["source_kind"]},
            ) as item:
                try:
                    if source["source_kind"] == "linkedin_reference":
                        url = linkedin_reference(source["source_url"])
                        repository.update_source(
                            source["source_id"], source_url=url, status="reference"
                        )
                except ProfileReadError as error:
                    item.fail(error.code)
                    failed(source, error)
        return {}

    @traced("candidate_profile", "extract_text")
    def extract(state):
        get_stream_writer()(
            {"stage": "extract_text", "label": "Reading your document and profile text…"}
        )
        for source in rows(state):
            if source["status"] != "pending":
                continue
            with span(
                "candidate_profile",
                "read_source",
                attributes={"source_id": source["source_id"], "source_kind": source["source_kind"]},
            ) as item:
                try:
                    text = (
                        read_document(source["input_data"], source["filename"])
                        if source["source_kind"] == "file"
                        else bounded_text(source["extracted_text"])
                    )
                    text = redact_credentials(text, stage="profile_input")
                    repository.update_source(
                        source["source_id"],
                        extracted_text=text,
                        input_data=None,
                        status="extracted",
                    )
                    item.attributes["characters"] = len(text)
                except Exception as error:
                    item.fail(
                        error.code if isinstance(error, ProfileReadError) else type(error).__name__
                    )
                    failed(source, error)
        return {}

    @traced("candidate_profile", "extract_facts")
    def facts(state):
        get_stream_writer()(
            {"stage": "extract_facts", "label": "Preparing your candidate profile…"}
        )
        for source in rows(state):
            if source["status"] != "extracted" and not source.get("retryable"):
                continue
            source["attempt_count"] += 1
            repository.update_source(source["source_id"], attempt_count=source["attempt_count"])
            with span(
                "candidate_profile",
                "source_facts",
                attributes={
                    "source_id": source["source_id"],
                    "source_kind": source["source_kind"],
                    "attempt": source["attempt_count"],
                },
            ) as item:
                try:
                    value = grounded_sections(
                        backend.extract_profile(source["extracted_text"]), source["extracted_text"]
                    )
                    repository.update_source(
                        source["source_id"],
                        facts=value,
                        status="ready",
                        retryable=False,
                        error_code=None,
                        error_message=None,
                    )
                except Exception as error:
                    item.fail(
                        error.code if isinstance(error, ProfileReadError) else type(error).__name__
                    )
                    failed(source, error)
        return {}

    @traced("candidate_profile", "merge_profile")
    def merge(state):
        get_stream_writer()({"stage": "merge_profile", "label": "Combining your supplied details…"})
        sources = rows(state)
        profile = merge_profile(sources, state.get("candidate_profile", {}).get("chat_updates", []))
        issues = [repository.summary(source) for source in sources if source["status"] == "failed"]
        ready = any(source["status"] == "ready" for source in sources) or bool(
            profile["chat_updates"]
        )
        status = (
            "partial" if issues and ready else "failed" if issues else "ready" if ready else "empty"
        )
        profile.update(status=status, sources=[repository.summary(source) for source in sources])
        return {"candidate_profile": profile, "profile_status": status, "profile_issues": issues}

    @traced("candidate_profile", "persist_profile")
    def persist(state):
        ids = state.get("profile_source_ids", [])
        profile_id = repository.save_profile(
            state["conversation_id"], state["request_id"], ids, state["candidate_profile"]
        )
        return {
            "profile_version": profile_id,
            "profile_input_fingerprint": snapshot_fingerprint(ids),
        }

    graph = StateGraph(ConversationState)
    for name, fn in (
        ("validate_sources", validate),
        ("extract_text", extract),
        ("extract_facts", facts),
        ("merge_profile", merge),
        ("persist_profile", persist),
    ):
        graph.add_node(name, fn)
    graph.add_edge(START, "validate_sources")
    for a, b in zip(
        ("validate_sources", "extract_text", "extract_facts", "merge_profile"),
        ("extract_text", "extract_facts", "merge_profile", "persist_profile"),
        strict=True,
    ):
        graph.add_edge(a, b)
    graph.add_edge("persist_profile", END)
    return graph.compile()
