from app.agents.backends.common import explicit_stop, latest_text


def advise(backend, state: dict) -> dict:
    if explicit_stop(latest_text(state)):
        label = "end"
    elif not state.get("messages"):
        label = "do_not_end"
    else:
        label = backend.exit_label(state["messages"])
    return {"exit_label": label, "advice": {"decision": label}}
