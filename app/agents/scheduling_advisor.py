from app.tools.scheduling import scheduling_tool


def advise(backend, repository, state: dict) -> dict:
    tool = scheduling_tool(repository, state)
    if state.get("selected_slot_id"):
        result = tool.invoke(
            {
                "date_text": None,
                "time_text": None,
                "period": None,
                "selected_slot_id": state["selected_slot_id"],
                "accepts_slot": True,
            }
        )
    else:
        result = backend.schedule(state, tool)
    if state.get("selected_slot_id"):
        result = {**result, "decision": "schedule"}
    if result.get("decision") not in {"schedule", "do_not_schedule"}:
        raise ValueError("Scheduling Advisor must return a scheduling decision")
    intent = dict(state.get("turn_intent", {}))
    if result["decision"] == "schedule":
        intent["scheduling_needed"] = True
    return {
        "schedule": result,
        "schedule_preferences": result.get("preferences", state.get("schedule_preferences", {})),
        "booking": result.get("booking", state.get("booking")),
        "turn_intent": intent,
        "advice": result,
    }
