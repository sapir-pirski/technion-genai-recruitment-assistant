"""Suggestion generation follows composition and cannot change operational decisions."""

import logging

from app.guardrails import check_followups
from app.observability.logging import log_event
from app.observability.tracing import span
from app.schemas import FollowupQuestions
from app.suggestions import (
    blocked_questions,
    fallback_questions,
    identified_questions,
    question_key,
    suggestion_context,
)


def suggest_followups(backend, state):
    with span("graph", "suggest_followups") as item:
        result = dict(state["result"])
        history = state.get("suggestion_history", [])
        questions, source, outcome, error_code = [], "none", "closed", None
        if result["conversation_status"] != "closed":
            try:
                if result.get("error"):
                    outcome = "answer_failed"
                    raise ValueError("Use fallback suggestions for a failed answer")
                outcome = "generation_failed"
                raw = backend.suggest_followups(suggestion_context(state))
                outcome = "invalid_output"
                generated = FollowupQuestions.model_validate(raw)
                questions = generated.questions
                check_followups(questions)
                if any(question_key(text) in blocked_questions(state) for text in questions):
                    outcome = "repeated_question"
                    raise ValueError("Suggestion already asked or recently offered")
                source, outcome = "model", "valid"
            except Exception as error:
                error_code = type(error).__name__
                source, questions = "fallback", fallback_questions(state)
        suggestions = identified_questions(state["conversation_id"], state["request_id"], questions)
        result.update(suggested_questions=suggestions, suggestion_source=source)
        item.attributes.update(
            suggestion_count=len(suggestions),
            suggestion_source=source,
            validation_outcome=outcome,
            error_code=error_code,
            suggestion_ids=[question["id"] for question in suggestions],
        )
        log_event(
            "followups_ready",
            level=logging.WARNING if source == "fallback" else logging.INFO,
            component="app.graph.followups",
            **item.attributes,
        )
        return {"result": result, "suggestion_history": [*history, *questions][-60:]}
