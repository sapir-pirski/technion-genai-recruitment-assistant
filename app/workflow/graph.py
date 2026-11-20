import logging

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.agents import exit_advisor, info_advisor, main_agent, scheduling_advisor
from app.guardrails import (
    GuardrailViolation,
    input_state,
    output_result,
    redact_credentials,
)
from app.observability.logging import log_event
from app.observability.tracing import span, traced
from app.workflow.state import ConversationState


def build_graph(backend, repository, retriever, checkpointer):
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
        return main_agent.compose(state)

    graph = StateGraph(ConversationState)
    graph.add_node("input_guard", validate_input)
    graph.add_node("output_guard", lambda state: output_result(state, repository))
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
    graph.add_edge("compose", "output_guard")
    graph.add_edge("output_guard", END)
    return graph.compile(checkpointer=checkpointer)
