import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import UTC
from pathlib import Path
from uuid import uuid4

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.observability.logging import log_event
from app.observability.tracing import span
from app.schemas import TurnResult
from app.storage.models import CandidateProfile, Conversation, ConversationMessage
from app.suggestions import STARTER_QUESTIONS


def utc_string(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class ConversationRepository:
    def __init__(self, engine, export_path):
        self.engine, self.export_path = engine, Path(export_path)

    @contextmanager
    def lock(self, conversation_id):
        # Session-level lock spans graph/checkpoint transactions and all application workers.
        with self.engine.connect() as connection:
            key = {"id": conversation_id}
            connection.execute(text("SELECT pg_advisory_lock(hashtextextended(:id, 732159))"), key)
            connection.commit()
            try:
                yield
            finally:
                connection.execute(
                    text("SELECT pg_advisory_unlock(hashtextextended(:id, 732159))"), key
                )
                connection.commit()

    def ensure(self, conversation_id, job_id, timezone, occurred_at):
        with Session(self.engine) as session, session.begin():
            row = session.get(Conversation, conversation_id)
            if row is None:
                session.add(
                    Conversation(
                        conversation_id=conversation_id,
                        job_id=job_id,
                        timezone=timezone,
                        started_at=occurred_at,
                    )
                )
            elif row.job_id != job_id or row.timezone != timezone:
                raise ValueError("Start a new conversation to change jobs or timezone")

    def completed_result(self, conversation_id, request_id):
        with Session(self.engine) as session:
            value = session.scalar(
                select(ConversationMessage.response_data).where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.request_id == request_id,
                    ConversationMessage.role == "assistant",
                )
            )
            return TurnResult(**value) if value else None

    def messages(self, conversation_id):
        """Read this conversation with each answer's original source/booking receipt."""
        with Session(self.engine) as session:
            return [
                {
                    "role": row.role,
                    "content": row.content,
                    "request_id": row.request_id,
                    "result": row.response_data or {},
                }
                for row in session.scalars(
                    select(ConversationMessage)
                    .where(ConversationMessage.conversation_id == conversation_id)
                    .order_by(ConversationMessage.sequence_no)
                )
            ]

    def _append(
        self, session, conversation_id, request_id, role, content, occurred_at, result=None
    ):
        existing = session.scalar(
            select(ConversationMessage).where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.request_id == request_id,
                ConversationMessage.role == role,
            )
        )
        if existing:
            if existing.content != content:
                raise ValueError("Request identifier already belongs to another message")
            return existing
        sequence = (
            session.scalar(
                select(func.max(ConversationMessage.sequence_no)).where(
                    ConversationMessage.conversation_id == conversation_id
                )
            )
            or 0
        ) + 1
        row = ConversationMessage(
            message_id=str(uuid4()),
            conversation_id=conversation_id,
            request_id=request_id,
            sequence_no=sequence,
            role=role,
            content=content,
            occurred_at=occurred_at,
            predicted_action=result.action if result else None,
            response_data=result.model_dump(mode="json") if result else None,
        )
        session.add(row)
        return row

    def user_message(
        self,
        conversation_id,
        request_id,
        content,
        occurred_at,
        *,
        context_inputs=None,
        profiles=None,
        suggestion=None,
    ):
        if content is not None:
            with Session(self.engine) as session, session.begin():
                row = self._append(
                    session, conversation_id, request_id, "user", content, occurred_at
                )
                if row.response_data is not None:
                    return row.response_data.get("context_source_ids")
                selection = suggestion.model_dump() if suggestion is not None else None
                if selection is not None:
                    self._validate_suggestion(session, conversation_id, content, selection)
                source_ids = None
                if context_inputs is not None:
                    source_ids = profiles.stage(
                        session, conversation_id, request_id, context_inputs, occurred_at
                    )
                row.response_data = {"context_source_ids": source_ids}
                if selection is not None:
                    row.response_data["suggestion"] = selection
            if selection is not None:
                with span("conversation", "select_suggestion", attributes=selection):
                    log_event("suggestion_selected", component="app.conversation", **selection)
            return source_ids
        return None

    def _validate_suggestion(self, session, conversation_id, content, selection):
        latest = session.scalar(
            select(ConversationMessage)
            .where(
                ConversationMessage.conversation_id == conversation_id,
                ConversationMessage.role == "assistant",
            )
            .order_by(ConversationMessage.sequence_no.desc())
            .limit(1)
        )
        if selection["answer_request_id"] is None and latest is None:
            choices = [question.model_dump() for question in STARTER_QUESTIONS]
        elif latest is not None and latest.request_id == selection["answer_request_id"]:
            choices = (latest.response_data or {}).get("suggested_questions", [])
        else:
            raise ValueError("This suggestion is no longer available")
        if not any(
            choice["id"] == selection["suggestion_id"] and choice["text"] == content
            for choice in choices
        ):
            raise ValueError("This suggestion does not belong to this answer")

    def complete(self, conversation_id, request_id, result, occurred_at):
        with Session(self.engine) as session, session.begin():
            self._append(
                session,
                conversation_id,
                request_id,
                "assistant",
                result.response,
                occurred_at,
                result,
            )
            row = session.get(Conversation, conversation_id)
            if row.status != "closed":
                row.status = result.conversation_status
                row.ended_at = occurred_at if result.conversation_status == "closed" else None

    def export_json(self):
        # Serialize snapshot reads AND replacement, so an older export cannot overwrite a newer one.
        with self.engine.begin() as connection:
            connection.execute(text("SELECT pg_advisory_xact_lock(732160)"))
            with Session(bind=connection) as session:
                records = {}
                # A single SELECT gives a consistent snapshot while other conversations write.
                rows = session.execute(
                    select(Conversation, ConversationMessage, CandidateProfile)
                    .outerjoin(
                        ConversationMessage,
                        Conversation.conversation_id == ConversationMessage.conversation_id,
                    )
                    .outerjoin(
                        CandidateProfile,
                        (Conversation.conversation_id == CandidateProfile.conversation_id)
                        & CandidateProfile.is_current.is_(True),
                    )
                    .order_by(
                        Conversation.started_at,
                        Conversation.conversation_id,
                        ConversationMessage.sequence_no,
                    )
                )
                for row, message, profile in rows:
                    record = records.setdefault(
                        row.conversation_id,
                        {
                            "conversation_id": row.conversation_id,
                            "candidate_phone": None,
                            "recruiter_phone": None,
                            "start_time_utc": utc_string(row.started_at),
                            "job_id": row.job_id,
                            "status": row.status,
                            "turns": [],
                        },
                    )
                    if profile is not None and "profile" not in record:
                        record["profile"] = {
                            "version": profile.profile_id,
                            "status": profile.status,
                            "created_at": utc_string(profile.created_at),
                            **{
                                key: value
                                for key, value in profile.data.items()
                                if key != "sources"
                            },
                        }
                        record["sources"] = profile.data.get("sources", [])
                    if message is None:
                        continue
                    turn = {
                        "turn_id": message.sequence_no,
                        "speaker": "candidate" if message.role == "user" else "recruiter",
                        "timestamp_utc": utc_string(message.occurred_at),
                        "text": message.content,
                        "label": None,
                        "predicted_action": message.predicted_action,
                        "request_id": message.request_id,
                    }
                    if message.role == "user" and message.response_data is not None:
                        turn["context_source_ids"] = (
                            message.response_data.get("context_source_ids", []) or []
                        )
                        if selection := message.response_data.get("suggestion"):
                            turn["suggestion"] = selection
                    elif message.role == "assistant":
                        receipt = message.response_data or {}
                        turn["suggested_questions"] = receipt.get("suggested_questions", [])
                        turn["suggestion_source"] = receipt.get("suggestion_source", "none")
                        turn["closure_reason"] = receipt.get("closure_reason")
                    record["turns"].append(turn)
                conversations = list(records.values())
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=self.export_path.parent,
                    prefix=".conversations-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    json.dump(conversations, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.export_path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return {"path": str(self.export_path), "conversations": len(conversations)}

    def try_export(self):
        try:
            return self.export_json()
        except Exception as error:
            log_event(
                "conversation_export_failed",
                level=logging.ERROR,
                component="app.storage",
                error_code=type(error).__name__,
            )
            return None
