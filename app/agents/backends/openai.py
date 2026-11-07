import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from app.agents.backends.common import OPERATION_PROMPTS, model_context, prompt_spec
from app.config import Settings
from app.guardrails import model_data
from app.observability.tracing import span
from app.schemas import (
    ChatProfileUpdate,
    FollowupQuestions,
    InfoAnswer,
    InfoDecision,
    JudgeResult,
    Plan,
    ProfileSections,
)


class OpenAIBackend:
    def __init__(self, settings: Settings):
        self.settings = settings
        options = dict(
            api_key=settings.require_api_key(),
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            max_completion_tokens=settings.max_output_tokens,
        )
        self.policy = settings.model_policy()
        self.prompts = {
            operation: prompt_spec(name, self.policy[operation]["model"])
            for operation, name in OPERATION_PROMPTS.items()
        }
        self.prompt_manifest = {
            operation: {key: value for key, value in spec.items() if key != "content"}
            for operation, spec in self.prompts.items()
        }

        def client(operation):
            policy = self.policy[operation]
            parameters = {**options, "model": policy["model"]}
            for key in ("reasoning_effort", "temperature"):
                if policy[key] is not None:
                    parameters[key] = policy[key]
            return ChatOpenAI(**parameters)

        self.chat = client("main")
        self.helper = client("profile_extraction")
        self.exit = client("exit")
        self.judge_client = client("judge")

    def _messages(self, operation, data):
        spec = self.prompts[operation]
        data = model_data(data, operation=operation)
        return [
            {"role": spec["instruction_role"], "content": spec["content"]},
            HumanMessage(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)),
        ]

    def plan(self, state: dict) -> Plan:
        return self._structured(
            Plan,
            "main",
            self._messages("main", model_context(state)),
        )

    def extract_profile(self, text: str) -> ProfileSections:
        return self._structured(
            ProfileSections,
            "profile_extraction",
            self._messages("profile_extraction", text),
        )

    def extract_chat_profile(self, context: dict) -> ChatProfileUpdate:
        return self._structured(
            ChatProfileUpdate, "chat_profile", self._messages("chat_profile", context)
        )

    def suggest_followups(self, context: dict) -> FollowupQuestions:
        return self._structured(
            FollowupQuestions,
            "followups",
            self._messages("followups", context),
        )

    def judge_quality(self, context: dict) -> JudgeResult:
        output_schema = JudgeResult.model_json_schema()
        references = ["question", "history", "answer", "followups"] + [
            item["evidence_id"]
            for key in ("sources", "calendar_receipts")
            for item in context.get(key, [])
        ]
        for name in ("JudgeRating", "JudgeIssue"):
            output_schema["$defs"][name]["properties"]["evidence_refs"]["items"]["enum"] = (
                references
            )
        return self._structured(
            JudgeResult,
            "judge",
            self._messages("judge", context),
            output_schema=output_schema,
        )

    def _structured(self, schema, operation, messages, *, output_schema=None):
        client = (
            self.judge_client
            if operation == "judge"
            else self.helper
            if operation in {"profile_extraction", "followups", "info_decision"}
            else self.chat
        )
        policy = self.policy[operation]
        # Set the model field before creating the include_raw runnable: its
        # parallel wrapper does not forward arbitrary invoke kwargs to the model.
        client = client.model_copy(
            update={
                "max_tokens": policy["max_completion_tokens"],
                "reasoning_effort": policy["reasoning_effort"],
            }
        )
        with span(
            "llm",
            operation,
            model=client.model_name,
            attributes={
                **self.prompt_manifest[operation],
                "reasoning_effort": policy["reasoning_effort"],
                "max_completion_tokens": policy["max_completion_tokens"],
            },
        ) as item:
            # A schema dictionary keeps SDK validation from hiding the raw usage response.
            response = client.with_structured_output(
                output_schema or schema.model_json_schema(),
                method="json_schema",
                strict=True,
                include_raw=True,
            ).invoke(messages)
            item.usage(response["raw"])
            if response.get("parsing_error"):
                raise response["parsing_error"]
            if response["parsed"] is None:
                raise ValueError("The model did not return the required structured response")
            return schema.model_validate(response["parsed"])

    def exit_label(self, history: list[dict]) -> str:
        with span(
            "llm",
            "exit",
            model=self.exit.model_name,
            attributes={
                **self.prompt_manifest["exit"],
                "reasoning_effort": self.policy["exit"]["reasoning_effort"],
                "max_completion_tokens": self.policy["exit"]["max_completion_tokens"],
            },
        ) as item:
            result = self.exit.invoke(
                self._messages("exit", {"history": history}),
                max_completion_tokens=self.policy["exit"]["max_completion_tokens"],
            )
            item.usage(result)
            label = result.content.strip() if isinstance(result.content, str) else ""
            if label not in ("end", "do_not_end"):
                raise ValueError("Exit Advisor returned an invalid label.")
            return label

    def schedule(self, state: dict, tool) -> dict:
        with span(
            "llm",
            "scheduling",
            model=self.chat.model_name,
            attributes=self.prompt_manifest["scheduling"],
        ) as item:
            response = self.chat.bind_tools(
                [tool], tool_choice="auto", parallel_tool_calls=False
            ).invoke(self._messages("scheduling", model_context(state, include_profile=False)))
            item.usage(response)
            if not response.tool_calls:
                if (
                    isinstance(response.content, str)
                    and response.content.strip() == "do_not_schedule"
                ):
                    return {"decision": "do_not_schedule", "status": "not_needed", "slots": []}
                raise ValueError("Scheduling must decline or call the calendar tool")
            if len(response.tool_calls) != 1 or response.tool_calls[0]["name"] != tool.name:
                raise ValueError("Scheduling Advisor must call the calendar tool exactly once.")
        return {**tool.invoke(response.tool_calls[0]["args"]), "decision": "schedule"}

    def info_decision(self, state):
        return self._structured(
            InfoDecision,
            "info_decision",
            self._messages(
                "info_decision",
                {
                    **model_context(state, include_profile=False),
                    "latest_candidate_message": state.get("messages", [])[-1]["content"]
                    if state.get("messages") and state["messages"][-1]["role"] == "user"
                    else "",
                },
            ),
        )

    def info(self, question: str, excerpts: list[dict], history=None) -> InfoAnswer:
        return self._structured(
            InfoAnswer,
            "info",
            self._messages(
                "info", {"question": question, "excerpts": excerpts, "history": history or []}
            ),
        )
