import json
import threading
from datetime import UTC, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.agents.backends.common import explicit_stop, prompt_manifest
from app.agents.backends.openai import OpenAIBackend
from app.config import Settings
from app.graph.builder import build_graph
from app.guardrails import input_text, redact_credentials
from app.ingestion.models import DEFAULT_JOB_ID
from app.ingestion.registry import JobRegistry
from app.observability.logging import configure_logging, log_errors, log_event
from app.observability.tracing import TraceRecorder, span
from app.profiles import PIPELINE_VERSION, content_hash
from app.rag.retriever import JobRetriever
from app.schemas import ProfileInputs, SuggestionSelection, TurnResult
from app.storage.conversation_repository import ConversationRepository
from app.storage.database import make_engine
from app.storage.profile_repository import ProfileRepository
from app.storage.schedule_repository import ScheduleRepository


class ChatService:
    def __init__(
        self,
        settings: Settings,
        *,
        backend=None,
        repository=None,
        retriever=None,
        checkpointer=None,
    ):
        self.settings = settings
        configure_logging(settings.log_level)
        self.registry = JobRegistry(settings)
        self.conversations = ConversationRepository(
            self.registry.engine, settings.conversation_export_path
        )
        self.traces = TraceRecorder(self.registry.engine)
        identity = content_hash(
            PIPELINE_VERSION
            + json.dumps(settings.model_policy()["profile_extraction"], sort_keys=True)
            + json.dumps(prompt_manifest(settings)["profile_extraction"], sort_keys=True)
        )[:24]
        self.profiles = ProfileRepository(self.registry.engine, PIPELINE_VERSION + ":" + identity)
        self.repository = repository or ScheduleRepository(make_engine(settings.database_url))
        self.repository.initialize()
        self.backend = backend if backend is not None else OpenAIBackend(settings)
        self.retriever = retriever or JobRetriever(settings)
        self._checkpoint_pool = None
        if checkpointer is None:
            self._checkpoint_pool = ConnectionPool(
                conninfo=settings.postgres_dsn,
                min_size=1,
                max_size=4,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                timeout=settings.request_timeout,
                open=True,
            )
            self._checkpoint_pool.wait(timeout=settings.request_timeout)
            checkpointer = PostgresSaver(self._checkpoint_pool)
            checkpointer.setup()
        self.graph = build_graph(
            self.backend, self.repository, self.retriever, checkpointer, self.profiles
        )
        # Serializes requests within a worker; DB constraints also protect across workers.
        self._lock = threading.RLock()

    def _config(self, conversation_id: str):
        return {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": 2 * self.settings.max_advisor_consultations + 14,
        }

    def state(self, conversation_id: str) -> dict:
        return self.graph.get_state(self._config(conversation_id)).values

    @log_errors("app.service")
    def turn(
        self,
        conversation_id: str,
        text: str | None = None,
        *,
        request_id: str | None = None,
        timezone: str | None = None,
        reference_time: datetime | None = None,
        selected_slot_id: int | None = None,
        job_id: str | None = None,
        context_inputs: ProfileInputs | dict | None = None,
        suggestion: SuggestionSelection | dict | None = None,
        progress=None,
    ) -> TurnResult:
        if len(conversation_id) > 64 or not conversation_id:
            raise ValueError("Invalid conversation identifier.")
        if context_inputs is not None:
            context_inputs = ProfileInputs.model_validate(context_inputs)
            if text is None:
                raise ValueError("Send a chat message to use your background information")
        if suggestion is not None:
            suggestion = SuggestionSelection.model_validate(suggestion)
            if text is None:
                raise ValueError("A suggestion must include its question")
        limit = 4000
        if text is not None and (not text.strip() or len(text) > limit):
            raise ValueError(f"Please enter a message between 1 and {limit:,} characters.")
        request_id = request_id or str(uuid4())
        if not request_id or len(request_id) > 64:
            raise ValueError("Invalid request identifier.")
        now = reference_time or datetime.now(UTC)
        if now.tzinfo is None:
            raise ValueError("reference_time must include a timezone.")
        with self._lock, self.conversations.lock(conversation_id):
            old = self.state(conversation_id)
            job_id = job_id or old.get("job_id") or DEFAULT_JOB_ID
            if old and old.get("job_id", DEFAULT_JOB_ID) != job_id:
                raise ValueError("Start a new conversation to change jobs")
            job = self.registry.get_job(job_id, collection=self.settings.index_collection)
            if not job or job["status"] != "active":
                raise ValueError(
                    "This job is not available; start a conversation for an active job"
                )
            job_context = {
                key: job[key]
                for key in ("job_id", "title", "company_name", "canonical_url", "locations")
            }
            job_context["scheduling_enabled"] = job["scheduling_enabled"]
            zone = timezone or old.get("timezone") or self.settings.business_timezone
            ZoneInfo(zone)
            if old and zone != old["timezone"]:
                raise ValueError("Start a new conversation to change timezone.")
            self.conversations.ensure(conversation_id, job_id, zone, now)
            completed = self.conversations.completed_result(conversation_id, request_id)
            if completed is not None:
                self.conversations.try_export()
                log_event(
                    "request_replayed",
                    component="app.service",
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
                return completed
            if request_id in old.get("completed_requests", {}):
                return self._recover_response(
                    conversation_id, request_id, text, now, old["completed_requests"][request_id]
                )
            unfinished = bool(self.graph.get_state(self._config(conversation_id)).next)
            if old.get("request_id") == request_id and old.get("result") and not unfinished:
                return self._recover_response(conversation_id, request_id, text, now, old["result"])
            if unfinished and old.get("request_id") != request_id:
                raise ValueError("Retry the pending request before sending another message")
            if old.get("conversation_status") == "closed" and not unfinished:
                return TurnResult(**old["result"])
            if text is None and old.get("result") and not unfinished:
                return TurnResult(**old["result"])
            if selected_slot_id and selected_slot_id not in {
                s["id"] for s in old.get("offered_slots", [])
            }:
                raise ValueError("Please choose an appointment offered in this conversation.")
            with self.traces.trace(
                conversation_id, request_id, attributes={"job_id": job_id, "input_kind": "chat"}
            ) as trace:
                try:
                    text = input_text(text)
                    if context_inputs is not None:
                        context_inputs = context_inputs.model_copy(
                            update={
                                key: redact_credentials(getattr(context_inputs, key), stage="input")
                                for key in (
                                    "introduction",
                                    "linkedin_text",
                                    "linkedin_url",
                                    "file_name",
                                )
                            }
                        )
                    messages = old.get("messages", [])
                    if text is not None and old.get("request_id") != request_id:
                        messages = [*messages, {"role": "user", "content": text.strip()}]
                    with span("conversation", "capture_inputs"):
                        source_ids = self.conversations.user_message(
                            conversation_id,
                            request_id,
                            text.strip() if text is not None else None,
                            now,
                            context_inputs=None if explicit_stop(text or "") else context_inputs,
                            profiles=self.profiles,
                            suggestion=suggestion,
                        )
                    payload = self._turn_payload(
                        old,
                        conversation_id,
                        request_id,
                        job_id,
                        job_context,
                        job,
                        messages,
                        now,
                        zone,
                        selected_slot_id,
                        source_ids,
                    )
                    return self._execute_turn(old, payload, trace, progress=progress)
                finally:
                    self.conversations.try_export()

    def _turn_payload(
        self,
        old,
        conversation_id,
        request_id,
        job_id,
        job_context,
        job,
        messages,
        now,
        zone,
        selected_slot_id,
        source_ids,
    ):
        return {
            "conversation_id": conversation_id,
            "job_id": job_id,
            "job_context": job_context,
            "scheduling_enabled": job["scheduling_enabled"],
            "request_id": request_id,
            "messages": messages,
            "reference_time": now.isoformat(),
            "timezone": zone,
            "selected_slot_id": selected_slot_id,
            "plan": {},
            "schedule": {},
            "schedule_preferences": old.get("schedule_preferences", {}),
            "info": {},
            "advisors": [],
            "exit_label": "do_not_end",
            "error": None,
            "result": {},
            "booking": self.repository.get_booking(conversation_id, zone, job_id),
            "advisor_results": [],
            "consulted_tasks": [],
            "consultation_count": 0,
            "max_consultations": self.settings.max_advisor_consultations,
            "turn_intent": {},
            "routing_stop_reason": None,
            "routing_history": [],
            "input_kind": "chat",
            "profile_source_ids": source_ids
            if source_ids is not None
            else old.get("profile_source_ids", []),
            "candidate_profile": old.get("candidate_profile", {}),
            "profile_version": old.get("profile_version"),
            "profile_status": old.get("profile_status", "empty"),
            "profile_input_fingerprint": old.get("profile_input_fingerprint"),
            "profile_issues": old.get("profile_issues", []),
            "suggestion_history": old.get("suggestion_history", []),
            "retrieved_source_ids": [],
            "answer_validated": False,
        }

    def _recover_response(self, conversation_id, request_id, text, now, receipt):
        result = TurnResult(**receipt)
        text = input_text(text)
        self.conversations.user_message(
            conversation_id, request_id, text.strip() if text is not None else None, now
        )
        self.conversations.complete(conversation_id, request_id, result, datetime.now(UTC))
        self.conversations.try_export()
        return result

    def _execute_turn(self, old, payload, trace, *, progress=None):
        conversation_id, request_id = payload["conversation_id"], payload["request_id"]
        config = self._config(conversation_id)
        resume = old.get("request_id") == request_id and bool(self.graph.get_state(config).next)
        if progress is None:
            result = self.graph.invoke(None if resume else payload, config, durability="sync")
        else:
            result = None
            answer_sent = False
            for namespace, mode, chunk in self.graph.stream(
                None if resume else payload,
                config,
                stream_mode=["values", "custom"],
                subgraphs=True,
                durability="sync",
            ):
                if mode == "custom" and isinstance(chunk, dict) and "label" in chunk:
                    self._report_progress(
                        progress, {"stage": chunk.get("stage"), "label": chunk["label"]}
                    )
                elif mode == "values" and not namespace:
                    result = chunk
                    if not answer_sent and chunk.get("result") and chunk.get("answer_validated"):
                        composed = TurnResult.model_validate(chunk["result"])
                        self._report_progress(
                            progress,
                            {
                                "stage": "answer_ready",
                                "label": "Finishing your reply…"
                                if composed.conversation_status == "closed"
                                else "Preparing follow-up questions…",
                                "request_id": request_id,
                                "result": composed.model_dump(),
                            },
                        )
                        answer_sent = True
            if result is None:
                raise ValueError("The conversation did not return a result")
        response = TurnResult(**result["result"])
        trace.attributes.update(
            action=response.action,
            closure_reason=response.closure_reason,
            advisors=response.advisors,
            source_ids=response.source_ids,
            consultation_count=result.get("consultation_count", 0),
            routing_history=result.get("routing_history", []),
            stop_reason=result.get("routing_stop_reason"),
            profile_version=result.get("profile_version"),
            profile_status=result.get("profile_status", "empty"),
            suggestion_count=len(response.suggested_questions),
            suggestion_source=response.suggestion_source,
        )
        if response.error:
            trace.fail(response.error)
        receipts = {**old.get("completed_requests", {}), request_id: result["result"]}
        receipts = dict(list(receipts.items())[-100:])
        self.graph.update_state(self._config(conversation_id), {"completed_requests": receipts})
        self.conversations.complete(conversation_id, request_id, response, datetime.now(UTC))
        return response

    @staticmethod
    def _report_progress(callback, event):
        try:
            callback(event)
        except Exception as error:
            log_event(
                "progress_delivery_failed",
                component="app.service",
                error_code=type(error).__name__,
            )

    def evaluate_prefix(
        self, history: list[dict], reference_time: datetime, timezone="UTC"
    ) -> TurnResult:
        """Stateless teacher-forced prediction; no gold response or future turns enter state."""
        isolated = build_graph(
            self.backend, self.repository, self.retriever, InMemorySaver(), followups=False
        )
        thread = str(uuid4())
        result = isolated.invoke(
            {
                "conversation_id": thread,
                "job_id": DEFAULT_JOB_ID,
                "scheduling_enabled": True,
                "request_id": str(uuid4()),
                "messages": history,
                "reference_time": reference_time.isoformat(),
                "timezone": timezone,
                "plan": {},
                "schedule": {},
                "info": {},
                "advisors": [],
                "offered_slots": [],
                "booking": None,
                "error": None,
                "selected_slot_id": None,
                "advisor_results": [],
                "consulted_tasks": [],
                "consultation_count": 0,
                "max_consultations": self.settings.max_advisor_consultations,
                "turn_intent": {},
                "routing_stop_reason": None,
                "routing_history": [],
                "input_kind": "chat",
            },
            self._config(thread),
        )
        return TurnResult(**result["result"])

    def close(self):
        if self._checkpoint_pool:
            self._checkpoint_pool.close()
        self.repository.engine.dispose()
        self.registry.close()
        if hasattr(self.retriever, "close"):
            self.retriever.close()
