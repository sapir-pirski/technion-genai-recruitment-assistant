"""Grounded chat facts: checkpoint extraction before persisting a profile version."""

import re

from langgraph.graph import END, START, StateGraph

from app.agents.backends.common import explicit_stop, latest_text
from app.graph.state import ConversationState
from app.observability.logging import log_event
from app.observability.tracing import span, traced
from app.profiles import (
    ProfileReadError,
    grounded_sections,
    merge_profile,
    snapshot_fingerprint,
    text_key,
)
from app.schemas import ChatProfileUpdate


def should_update_chat(state):
    text = latest_text(state)
    if not text or explicit_stop(text):
        return False
    declaration = re.search(
        r"\b(?:i(?:'m|'ve| am| have| use| know| worked| built| studied| graduated)|my (?:name|skills|experience|background|project)|(?:name|skills|experience)\s*:)",
        text,
        re.I,
    )
    background = re.search(
        r"\b(?:python|developer|skills?|experience|years?|projects?|worked|built|studied|graduated|name|education)\b",
        text,
        re.I,
    )
    short_experience = re.fullmatch(
        r"\s*\d+(?:\.\d+)?\s*(?:years?|months?)(?: of Python experience)?[.!]?\s*", text, re.I
    )
    affirmation = re.fullmatch(
        r"\s*(?:yes|correct|that is correct|that's correct|confirmed)[.!]?\s*", text, re.I
    )
    confirmation_statement = re.search(r"\b(?:i confirm|is correct|is right)\b", text, re.I)
    return bool(
        ((declaration or confirmation_statement) and background)
        or short_experience
        or (affirmation and state.get("candidate_profile", {}).get("conflicts"))
    )


def build_chat_profile(backend, repository):
    @traced("candidate_profile", "extract_chat_update")
    def extract(state):
        message = repository.chat_message(state["conversation_id"], state["request_id"])
        old = state.get("candidate_profile", {})
        if message is None or any(
            entry["message_id"] == message["message_id"] for entry in old.get("chat_updates", [])
        ):
            return {"chat_profile_update": None}
        known = [
            item for item in old.get("evidence", []) if item.get("status", "active") == "active"
        ][-40:]
        # Only facts with a stable identity can be revised.
        known = [item for item in known if item.get("fact_id")]
        last_question = next(
            (
                item["content"]
                for item in reversed(state.get("messages", [])[:-1])
                if item["role"] == "assistant"
            ),
            "",
        )
        with span(
            "candidate_profile",
            "chat_facts",
            attributes={"message_id": message["message_id"], "known_fact_count": len(known)},
        ) as item:
            try:
                update = ChatProfileUpdate.model_validate(
                    backend.extract_chat_profile(
                        {
                            "latest_message": message["content"],
                            "prior_assistant_question": last_question[-1500:],
                            "known_facts": [
                                {key: fact[key] for key in ("fact_id", "field", "quote")}
                                for fact in known
                            ],
                        }
                    )
                )
                facts = grounded_sections(
                    update.facts,
                    message["content"],
                    allow_terminal_punctuation=True,
                    drop_ungrounded=True,
                )
                item.attributes["rejected_fact_count"] = sum(
                    len(value) if isinstance(value, list) else int(bool(value))
                    for value in update.facts.model_dump().values()
                ) - sum(
                    len(value) if isinstance(value, list) else int(bool(value))
                    for value in facts.values()
                )
                by_id = {fact["fact_id"]: fact for fact in known}
                # Invalid model-proposed revisions must not discard valid new facts.
                # Preserve the earlier claim and let Main clarify the resulting conflict.
                rejected = 0
                superseded = []
                confirmed = []
                correction = re.search(
                    r"\b(?:actually|correction|correct|instead|rather|not|meant|replace|updated)\b",
                    message["content"],
                    re.I,
                )
                for fact_id in update.superseded_fact_ids:
                    if (
                        fact_id not in by_id
                        or not correction
                        or not facts.get(by_id[fact_id]["field"])
                    ):
                        rejected += 1
                        continue
                    superseded.append(fact_id)
                for fact_id in update.confirmed_fact_ids:
                    if fact_id not in by_id:
                        rejected += 1
                        continue
                    quote = text_key(by_id[fact_id]["quote"])
                    positive = text_key(
                        re.split(
                            r"\b(?:not|instead|rather than)\b",
                            message["content"],
                            maxsplit=1,
                            flags=re.I,
                        )[0]
                    )
                    short_yes = re.fullmatch(
                        r"(?:yes|correct|that is correct|that's correct|confirmed)[.!]?", positive
                    )
                    mentioned = [
                        fact for fact in known if text_key(fact["quote"]) in text_key(last_question)
                    ]
                    known_years = set(re.findall(r"(\d+(?:\.\d+)?)\s*years?", quote))
                    affirmed_years = set(re.findall(r"(\d+(?:\.\d+)?)\s*years?", positive))
                    same_years = (
                        by_id[fact_id]["field"] == "experience"
                        and "python" in quote
                        and "python" in positive
                        and len(known_years) == len(affirmed_years) == 1
                        and known_years == affirmed_years
                    )
                    if "?" in message["content"] or not (
                        (
                            (quote in positive or same_years)
                            and re.search(r"\b(?:yes|confirm|correct|right)\b", positive)
                        )
                        or (
                            short_yes and len(mentioned) == 1 and mentioned[0]["fact_id"] == fact_id
                        )
                    ):
                        rejected += 1
                        continue
                    confirmed.append(fact_id)
                item.attributes["rejected_revision_count"] = rejected
                if not any(facts.values()) and not (superseded or confirmed):
                    return {"chat_profile_update": None}
                result = {
                    "message_id": message["message_id"],
                    "request_id": state["request_id"],
                    "facts": facts,
                    "superseded_fact_ids": superseded,
                    "confirmed_fact_ids": confirmed,
                    "context_subject": "python_experience"
                    if re.search(r"Python.*experience|experience.*Python", last_question, re.I)
                    else None,
                }
                item.attributes.update(
                    fact_count=sum(
                        len(value) if isinstance(value, list) else int(bool(value))
                        for value in facts.values()
                    ),
                    superseded_count=len(superseded),
                    confirmed_count=len(confirmed),
                )
                return {"chat_profile_update": result}
            except Exception as error:
                code = error.code if isinstance(error, ProfileReadError) else type(error).__name__
                item.fail(code)
                log_event(
                    "chat_profile_update_failed",
                    component="app.profiles",
                    error_code=code,
                )
                return {"chat_profile_update": None}

    @traced("candidate_profile", "persist_chat_update")
    def persist(state):
        update = state.get("chat_profile_update")
        if not update:
            return {}
        ids = state.get("profile_source_ids", [])
        sources = repository.sources(state["conversation_id"], ids)
        previous = state.get("candidate_profile", {})
        updates = [
            entry
            for entry in previous.get("chat_updates", [])
            if entry["message_id"] != update["message_id"]
        ] + [update]
        profile = merge_profile(sources, updates)
        issues = state.get("profile_issues", [])
        status = "partial" if issues else "ready"
        profile.update(status=status, sources=[repository.summary(source) for source in sources])
        profile_id = repository.save_profile(
            state["conversation_id"], state["request_id"], ids, profile
        )
        return {
            "candidate_profile": profile,
            "profile_status": status,
            "profile_version": profile_id,
            "profile_input_fingerprint": snapshot_fingerprint(ids),
            "chat_profile_update": None,
        }

    graph = StateGraph(ConversationState)
    graph.add_node("extract_chat_update", extract)
    graph.add_node("persist_chat_update", persist)
    graph.add_edge(START, "extract_chat_update")
    graph.add_edge("extract_chat_update", "persist_chat_update")
    graph.add_edge("persist_chat_update", END)
    return graph.compile()
