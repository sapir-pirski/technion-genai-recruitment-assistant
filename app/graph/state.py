from typing import TypedDict


class ConversationState(TypedDict, total=False):
    job_id: str
    job_context: dict
    scheduling_enabled: bool
    conversation_id: str
    request_id: str
    messages: list[dict]
    candidate_profile: dict
    profile_source_ids: list[str]
    profile_version: str | None
    profile_input_fingerprint: str | None
    profile_status: str
    profile_issues: list[dict]
    chat_profile_update: dict | None
    timezone: str
    reference_time: str
    selected_slot_id: int | None
    offered_slots: list[dict]
    booking: dict | None
    plan: dict
    exit_label: str
    schedule: dict
    info: dict
    advisors: list[str]
    error: str | None
    result: dict
    conversation_status: str
    completed_requests: dict
    declined_slot_ids: list[int]
    advisor_results: list[dict]
    consultation_count: int
    max_consultations: int
    consulted_tasks: list[str]
    turn_intent: dict
    routing_stop_reason: str | None
    routing_history: list[dict]
    input_kind: str
    suggestion_history: list[str]
    retrieved_source_ids: list[str]
    answer_validated: bool
