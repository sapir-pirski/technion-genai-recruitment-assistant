import logging

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents import exit_advisor, info_advisor, main_agent, scheduling_advisor
from app.agents.backends.common import explicit_stop, latest_text
from app.graph.chat_profile import build_chat_profile, should_update_chat
from app.graph.followups import suggest_followups
from app.graph.profile_ingestion import build_profile_ingestion, should_ingest
from app.graph.state import ConversationState
from app.guardrails import (
    GuardrailViolation,
    check_followups,
    input_state,
    output_result,
    record,
    redact_credentials,
)
from app.observability.logging import log_event
from app.observability.tracing import span, traced
from app.schemas import FollowupQuestions
from app.suggestions import fallback_questions, identified_questions


def build_graph(backend, repository, retriever, checkpointer, profiles=None, *, followups=True):
    def validate_input(state):
        try:
            return input_state(state)
        except GuardrailViolation:
            return {"error": "GuardrailViolation"}

    def main(state):
        try:
            with span("agent", "main") as item:
                updates = redact_credentials(
                    main_agent.make_plan(backend, state), stage="model_output"
                )
                item.attributes.update(
                    decision=updates["plan"]["next_advisor"],
                    consultation_count=state.get("consultation_count", 0),
                    stop_reason=updates["routing_stop_reason"],
                )
                log_event("main_routed", component="app.graph", **item.attributes)
                return updates
        except Exception as error:
            log_event(
                "main_failed",
                level=logging.WARNING,
                component="app.graph",
                error_code=type(error).__name__,
            )
            return {"error": type(error).__name__}

    def advisor(name, fn):
        def consult(state):
            try:
                labels = {
                    "exit": "Reviewing your message…",
                    "info": "Checking the job description…",
                    "scheduling": "Checking interview times…",
                }
                get_stream_writer()({"stage": name, "label": labels[name]})
                with span("agent", name) as item:
                    updates = redact_credentials(fn(state), stage="model_output")
                    advice = updates.pop("advice")
                    allowed = {
                        "exit": {"end", "do_not_end"},
                        "info": {"info_needed", "info_not_needed"},
                        "scheduling": {"schedule", "do_not_schedule"},
                    }
                    if advice.get("decision") not in allowed[name]:
                        raise ValueError("Advisor returned an invalid decision")
                    item.attributes["decision"] = advice["decision"]
                    log_event(
                        "advisor_returned",
                        component=f"app.advisor.{name}",
                        decision=advice["decision"],
                    )
            except Exception as error:
                updates = {"error": type(error).__name__}
                advice = {"decision": "failed", "error_code": type(error).__name__}
                log_event(
                    "advisor_failed",
                    level=logging.WARNING,
                    component=f"app.advisor.{name}",
                    error_code=type(error).__name__,
                )
            return {
                **updates,
                "advisors": [*state.get("advisors", []), name],
                "advisor_results": [*state.get("advisor_results", []), {"advisor": name, **advice}],
                "consultation_count": state.get("consultation_count", 0) + 1,
                "consulted_tasks": [
                    *state.get("consulted_tasks", []),
                    main_agent.task_fingerprint(state, state["plan"]),
                ],
            }

        return consult

    @traced("agent", "compose")
    def finalize(state):
        if not state.get("plan"):
            state = {**state, "plan": main_agent.fallback_plan(state).model_dump()}
        composed = main_agent.compose(state)
        return output_result({**state, **composed}, repository)

    def validate_output(state):
        checked = output_result(state, repository)
        result = checked["result"]
        try:
            questions = [item["text"] for item in result["suggested_questions"]]
            if result["conversation_status"] == "closed" or not followups:
                result.update(suggested_questions=[], suggestion_source="none")
            else:
                FollowupQuestions(questions=questions)
                check_followups(questions)
        except ValueError:
            record("suggestion_shape", "output", "clarify", "invalid_suggestions")
            questions = fallback_questions({**state, **checked})
            result.update(
                suggested_questions=identified_questions(
                    state["conversation_id"], state["request_id"], questions
                ),
                suggestion_source="fallback",
            )
            checked["suggestion_history"] = [*state.get("suggestion_history", []), *questions][-60:]
        return checked

    graph = StateGraph(ConversationState)
    graph.add_node("input_guard", validate_input)
    graph.add_node("output_guard", validate_output)
    graph.add_edge(START, "input_guard")
    graph.add_node("main", main)
    graph.add_node(
        "exit_advisor", advisor("exit", lambda state: exit_advisor.advise(backend, state))
    )
    graph.add_node(
        "scheduling_advisor",
        advisor("scheduling", lambda state: scheduling_advisor.advise(backend, repository, state)),
    )
    graph.add_node(
        "info_advisor",
        advisor("info", lambda state: info_advisor.advise(backend, retriever, state)),
    )
    graph.add_node("compose", finalize)
    if profiles is not None:
        graph.add_node("profile_ingestion", build_profile_ingestion(backend, profiles))
        graph.add_node("chat_profile", build_chat_profile(backend, profiles))

        def next_profile_step(state):
            return (
                "chat_profile" if should_update_chat(state) and not state.get("error") else "main"
            )

        graph.add_conditional_edges(
            "input_guard",
            lambda state: (
                "main"
                if state.get("error")
                else "profile_ingestion"
                if (not explicit_stop(latest_text(state)) and should_ingest(state, profiles))
                else next_profile_step(state)
            ),
            {
                "profile_ingestion": "profile_ingestion",
                "chat_profile": "chat_profile",
                "main": "main",
            },
        )
        graph.add_conditional_edges(
            "profile_ingestion", next_profile_step, {"chat_profile": "chat_profile", "main": "main"}
        )
        graph.add_edge("chat_profile", "main")
    else:
        graph.add_edge("input_guard", "main")
    graph.add_conditional_edges(
        "main",
        lambda state: "respond" if state.get("error") else state["plan"]["next_advisor"],
        {
            "exit": "exit_advisor",
            "info": "info_advisor",
            "scheduling": "scheduling_advisor",
            "respond": "compose",
        },
    )
    for name in ("exit_advisor", "scheduling_advisor", "info_advisor"):
        graph.add_edge(name, "main")
    if followups:
        graph.add_node("suggest_followups", lambda state: suggest_followups(backend, state))
        graph.add_edge("compose", "suggest_followups")
        graph.add_edge("suggest_followups", "output_guard")
    else:
        graph.add_edge("compose", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile(checkpointer=checkpointer)
