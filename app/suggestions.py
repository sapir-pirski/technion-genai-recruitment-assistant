"""Stable starters and bounded context for candidate-initiated follow-up questions."""

import hashlib
import re

from app.schemas import SuggestedQuestion

STARTER_QUESTIONS = (
    SuggestedQuestion(id="starter-skills", text="What skills are required for this role?"),
    SuggestedQuestion(id="starter-responsibilities", text="What are the main responsibilities?"),
    SuggestedQuestion(id="starter-interview", text="How can I arrange an interview?"),
)


def question_key(text):
    return re.sub(r"\W+", " ", text).strip().casefold()


def suggestion_context(state):
    result = state["result"]
    return {
        "job": state.get("job_context", {}),
        "history": [
            {"role": message["role"], "content": message["content"][:2000]}
            for message in state.get("messages", [])[-12:]
        ],
        "latest_answer": result["response"][:6000],
        "previous_suggestions": state.get("suggestion_history", [])[-30:],
        "scheduling_enabled": state.get("scheduling_enabled", True),
        "booking_confirmed": bool(result.get("booking")),
        "appointments_offered": bool(result.get("offered_slots")),
    }


def blocked_questions(state):
    values = [
        message["content"] for message in state.get("messages", []) if message["role"] == "user"
    ]
    return {question_key(text) for text in [*values, *state.get("suggestion_history", [])[-30:]]}


def fallback_questions(state):
    """Capability-aware questions only; never fabricate facts or accept an appointment."""
    result = state["result"]
    if result.get("booking"):
        priority = [
            "What should I prepare for my confirmed interview?",
            "What does the interview process involve?",
            "Which parts of my experience should I discuss in the interview?",
        ]
    elif result.get("offered_slots"):
        priority = [
            "Can I see interview times on another day?",
            "How are these interview times shown in my timezone?",
            "What should I prepare for the interview?",
        ]
    else:
        latest_user = next(
            (
                message["content"]
                for message in reversed(state.get("messages", []))
                if message["role"] == "user"
            ),
            "",
        )
        topic = (
            state.get("turn_intent", {}).get("info_question") or latest_user or result["response"]
        ).casefold()
        if re.search(r"skill|require|experience|qualif", topic):
            priority = [
                "Which skills would I use most often in this role?",
                "How could my projects demonstrate these skills?",
                "How can I learn more about the interview process?",
            ]
        elif re.search(r"responsib|work|project|duties", topic):
            priority = [
                "What kinds of projects would I work on?",
                "How would I collaborate with the team?",
                "Which skills support these responsibilities?",
            ]
        else:
            priority = [
                "Which parts of my background are relevant to this role?",
                "What else would you like to know about my experience?",
                "What should I know about the day-to-day work?",
            ]
    process = (
        "What are the next steps for arranging an interview?"
        if state.get("scheduling_enabled", True) and not result.get("booking")
        else "How can I contact the employer about the next steps?"
    )
    candidates = list(
        dict.fromkeys(
            [
                *priority,
                process,
                "What can you tell me about the team?",
                "What technologies are mentioned in the job description?",
                "What does the description say about the work location?",
                "How could I explain a relevant Python project?",
                "What can you tell me about learning opportunities?",
                "Which details should I clarify with the recruiter?",
                "Can you explain the role requirements in more detail?",
                "What would help me understand whether this role suits me?",
            ]
        )
    )
    blocked = blocked_questions(state)
    fresh = [text for text in candidates if question_key(text) not in blocked]
    # A long outage can exhaust the fallback pool; prefer older suggestions over no response.
    return (fresh + [text for text in candidates if text not in fresh])[:3]


def identified_questions(conversation_id, request_id, questions):
    return [
        SuggestedQuestion(
            id="q_"
            + hashlib.sha256(f"{conversation_id}\0{request_id}\0{text}".encode()).hexdigest()[:24],
            text=text,
        ).model_dump()
        for text in questions
    ]
