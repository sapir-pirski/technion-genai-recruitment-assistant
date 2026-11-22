import hashlib
import json
from unittest.mock import Mock, patch

import httpx
import pytest
from langchain_openai import ChatOpenAI

from app.agents.backends.common import OPERATION_PROMPTS, prompt_spec
from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from app.tools.scheduling import scheduling_tool


def test_function_call_executes_validated_calendar_tool(repository, now):
    seen = []

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_calendar",
                                    "type": "function",
                                    "function": {
                                        "name": "manage_interview",
                                        "arguments": json.dumps(
                                            {
                                                "date_text": "today",
                                                "time_text": None,
                                                "period": None,
                                                "selected_slot_id": None,
                                                "accepts_slot": False,
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    settings = Settings(_env_file=None, openai_api_key="test-not-a-real-key")
    backend = OpenAIBackend(settings)
    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        backend.chat = ChatOpenAI(
            model=settings.chat_model, api_key="test-not-a-real-key", http_client=client
        )
        state = {
            "messages": [{"role": "user", "content": "Any appointments today?"}],
            "reference_time": now.isoformat(),
            "timezone": "UTC",
            "conversation_id": "mock",
            "request_id": "r1",
            "offered_slots": [],
            "candidate_profile": {"private_profile_details": "not needed for calendar queries"},
        }
        result = backend.schedule(state, scheduling_tool(repository, state))
    assert result["status"] == "offered"
    assert len(result["slots"]) == 3
    assert result["decision"] == "schedule"
    assert seen[0]["tool_choice"] == "auto"
    assert seen[0]["tools"][0]["function"]["name"] == "manage_interview"
    assert seen[0]["parallel_tool_calls"] is False
    assert seen[0]["messages"][0]["role"] == "system"
    assert (
        seen[0]["messages"][0]["content"]
        == prompt_spec("scheduling", settings.chat_model)["content"]
    )
    assert "candidate_profile" not in json.loads(seen[0]["messages"][1]["content"])


def test_scheduling_can_decline_without_querying_calendar(repository, now):
    content = "do_not_schedule"
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }
    backend = OpenAIBackend(Settings(_env_file=None, openai_api_key="test-only-key"))
    repository.available = Mock()
    repository.book = Mock()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    ) as client:
        backend.chat = ChatOpenAI(model="gpt-4o-mini", api_key="test-only-key", http_client=client)
        state = {
            "messages": [{"role": "user", "content": "I use Python."}],
            "reference_time": now.isoformat(),
            "timezone": "UTC",
            "conversation_id": "no-calendar",
            "request_id": "request",
        }
        assert (
            backend.schedule(state, scheduling_tool(repository, state))["decision"]
            == "do_not_schedule"
        )
    repository.available.assert_not_called()
    repository.book.assert_not_called()


@pytest.mark.parametrize(
    "operation,output",
    [
        ("main", None),
        (
            "profile_extraction",
            {"name": None, "experience": [], "skills": ["Python"], "education": [], "projects": []},
        ),
    ],
)
def test_operation_selects_model_and_sends_compatible_bounded_parameters(
    now, backend, operation, output
):
    settings = Settings(_env_file=None, openai_api_key="test-only-key")
    seen = []
    content = json.dumps(output or backend.plan.return_value.model_dump())

    def handle(request):
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-policy",
                "object": "chat.completion",
                "created": 1,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with patch(
            "app.agents.backends.openai.ChatOpenAI",
            side_effect=lambda **options: ChatOpenAI(http_client=client, **options),
        ):
            adapter = OpenAIBackend(settings)
        state = {"messages": [], "reference_time": now.isoformat(), "timezone": "UTC"}
        calls = {
            "main": lambda: adapter.plan(state),
            "profile_extraction": lambda: adapter.extract_profile("Python"),
        }
        calls[operation]()
        if operation == "main":
            seen.clear()
            # Chat revisions use the existing non-reasoning budget model and system prompt.
            content = json.dumps(
                {
                    "facts": {
                        "name": None,
                        "experience": [],
                        "skills": [],
                        "education": [],
                        "projects": [],
                    },
                    "superseded_fact_ids": [],
                    "confirmed_fact_ids": [],
                }
            )
            adapter.extract_chat_profile({"latest_message": "I use Python."})
            assert seen[0]["model"] == settings.model_for("chat_profile") == settings.chat_model
            assert seen[0]["messages"][0]["role"] == "system"
            assert seen[0]["max_completion_tokens"] == settings.max_output_tokens
            assert "reasoning_effort" not in seen[0]
            seen.clear()
            content = json.dumps({"decision": "info_not_needed", "question": None})
            adapter.info_decision(state)
            assert seen[0]["reasoning_effort"] == "minimal"
            assert seen[0]["max_completion_tokens"] == 1024
            seen.clear()
            content = json.dumps(backend.plan.return_value.model_dump())
            adapter.plan(state)
    assert len(seen) == 1
    request = seen[0]
    policy = settings.model_policy()[operation]
    assert request["model"] == policy["model"]
    assert request["max_completion_tokens"] == policy["max_completion_tokens"]
    assert "max_tokens" not in request
    spec = prompt_spec(OPERATION_PROMPTS[operation], policy["model"])
    assert request["messages"][0] == {"role": spec["instruction_role"], "content": spec["content"]}
    assert request["messages"][1]["role"] == "user"
    assert (
        adapter.prompt_manifest[operation]["prompt_sha256"]
        == hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest()
    )
    if policy["model"] == settings.helper_model:
        assert request["reasoning_effort"] == policy["reasoning_effort"]
        assert "temperature" not in request
    else:
        assert request["temperature"] == 0
        assert "reasoning_effort" not in request
