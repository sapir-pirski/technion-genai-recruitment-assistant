import re

from app.agents.backends.common import latest_text
from app.guardrails import check_sources, redact_credentials, reject
from app.observability.tracing import span
from app.schemas import InfoDecision


def advise(backend, retriever, state: dict) -> dict:
    decision = backend.info_decision(state)
    question = state.get("turn_intent", {}).get("info_question")
    latest = latest_text(state)
    explicit_questions = re.findall(r"[^.!?\n]*\?", latest)
    if (
        decision.decision == "info_not_needed"
        and question
        and explicit_questions
        and not any(
            item["advisor"] == "info" and item.get("question") == question
            for item in state.get("advisor_results", [])
        )
    ):
        # Preserve the candidate's actual question when the helper mistakes
        # a scheduling receipt or an older answer for an answer to this new request.
        decision = InfoDecision(
            decision="info_needed", question=" ".join(explicit_questions).strip()
        )
        with span(
            "guardrail", "info_request", attributes={"reason_code": "explicit_question_preserved"}
        ):
            pass
    if decision.decision == "info_not_needed":
        return {"advice": decision.model_dump()}
    question = decision.question
    question = redact_credentials(question, stage="retrieval")
    with span("retrieval", "job_search") as item:
        excerpts = retriever.search(question, job_id=state.get("job_id"))
        if any(
            excerpt.get("job_id") is not None and excerpt["job_id"] != state.get("job_id")
            for excerpt in excerpts
        ):
            reject(
                "job_scope",
                "retrieval",
                "different_job",
                "The retrieved description belongs to another job.",
            )
        excerpts = redact_credentials(excerpts, stage="retrieval")
        item.attributes["source_ids"] = [excerpt["id"] for excerpt in excerpts]
    if not excerpts:
        info = {
            "answer": "The job information isn't available right now. A recruiter can clarify that.",
            "source_ids": [],
            "supported": False,
        }
    else:
        result = backend.info(question, excerpts, state.get("messages", []))
        valid = {item["id"] for item in excerpts}
        if result.supported:
            check_sources(result.source_ids, valid)
        if not result.supported:
            result.answer = "The job description doesn't specify that. A recruiter can clarify it during your interview."
            result.source_ids = []
        info = result.model_dump()
    advice = {**decision.model_dump(), **info}
    previous = state.get("info", {})
    if previous and previous.get("answer") != info["answer"]:
        info = {
            **info,
            "answer": previous["answer"] + "\n\n" + info["answer"],
            "source_ids": list(
                dict.fromkeys([*previous.get("source_ids", []), *info["source_ids"]])
            ),
        }
    return {
        "retrieved_source_ids": list(
            dict.fromkeys(
                [*state.get("retrieved_source_ids", []), *[item["id"] for item in excerpts]]
            )
        ),
        "info": info,
        "turn_intent": {**state.get("turn_intent", {}), "info_question": question},
        "advice": advice,
    }
