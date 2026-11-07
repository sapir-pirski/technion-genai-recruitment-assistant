"""Immutable request snapshots and conversation-scoped candidate source processing."""

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.profiles import (
    MAX_FILE_BYTES,
    MAX_PROFILE_CHARS,
    MEDIA_TYPES,
    content_hash,
    safe_filename,
    snapshot_fingerprint,
)
from app.storage.models import CandidateProfile, CandidateSource, ConversationMessage


def utcnow():
    return datetime.now(UTC)


class ProfileRepository:
    def __init__(self, engine, pipeline_version):
        self.engine, self.pipeline_version = engine, pipeline_version

    def stage(self, session, conversation_id, request_id, inputs, occurred_at):
        values = []
        for kind, value in (
            ("introduction", inputs.introduction),
            ("linkedin_text", inputs.linkedin_text),
        ):
            if value:
                values.append(
                    {"source_kind": kind, "extracted_text": value, "content": value.encode()}
                )
        if inputs.linkedin_url:
            values.append(
                {
                    "source_kind": "linkedin_reference",
                    "source_url": inputs.linkedin_url,
                    "content": inputs.linkedin_url.encode(),
                }
            )
        if inputs.file_data is not None:
            values.append(
                {
                    "source_kind": "file",
                    "filename": safe_filename(inputs.file_name),
                    "content": inputs.file_data,
                }
            )
        ids = []
        for value in values:
            content = value.pop("content")
            kind = value["source_kind"]
            suffix = Path(value.get("filename", "")).suffix.casefold()
            sha = content_hash(content)
            identity = f"{conversation_id}:{kind}:{sha}:{suffix}:{self.pipeline_version}"
            source_id = content_hash(identity)
            ids.append(source_id)
            if session.get(CandidateSource, source_id):
                continue
            error = None
            if kind == "file" and (not content or len(content) > MAX_FILE_BYTES):
                error = ("FileSizeLimit", "Upload a document no larger than 5 MB.")
            elif kind in {"introduction", "linkedin_text"} and len(value["extracted_text"]) > (
                4000 if kind == "introduction" else MAX_PROFILE_CHARS
            ):
                error = ("TextLimitExceeded", "Shorten this source before sending another message.")
            elif kind == "linkedin_reference" and len(value["source_url"]) > 500:
                error = (
                    "InvalidLinkedInURL",
                    "Use a LinkedIn profile URL no longer than 500 characters.",
                )
            session.add(
                CandidateSource(
                    source_id=source_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    content_sha256=sha,
                    pipeline_version=self.pipeline_version,
                    byte_count=len(content),
                    media_type=MEDIA_TYPES.get(suffix) if kind == "file" else "text/plain",
                    input_data=content if kind == "file" and not error else None,
                    status="failed" if error else "pending",
                    error_code=error[0] if error else None,
                    error_message=error[1] if error else None,
                    created_at=occurred_at,
                    **{
                        key: item
                        for key, item in value.items()
                        if not error or key not in {"source_url", "extracted_text"}
                    },
                )
            )
        session.flush()
        return ids

    def sources(self, conversation_id, source_ids):
        with Session(self.engine) as session:
            rows = {
                row.source_id: {
                    column.name: getattr(row, column.name)
                    for column in CandidateSource.__table__.columns
                }
                for row in session.scalars(
                    select(CandidateSource).where(
                        CandidateSource.conversation_id == conversation_id,
                        CandidateSource.source_id.in_(source_ids),
                    )
                )
            }
        if set(rows) != set(source_ids):
            raise ValueError("Candidate source does not belong to this conversation")
        return [rows[key] for key in source_ids]

    def update_source(self, source_id, **values):
        with Session(self.engine) as session, session.begin():
            row = session.get(CandidateSource, source_id)
            for key, value in values.items():
                setattr(row, key, value)
            if values.get("status") in {"ready", "reference", "failed"}:
                row.processed_at = utcnow()
                row.input_data = None

    def chat_message(self, conversation_id, request_id):
        with Session(self.engine) as session:
            row = session.scalar(
                select(ConversationMessage).where(
                    ConversationMessage.conversation_id == conversation_id,
                    ConversationMessage.request_id == request_id,
                    ConversationMessage.role == "user",
                )
            )
            return {"message_id": row.message_id, "content": row.content} if row else None

    def save_profile(self, conversation_id, request_id, source_ids, data):
        fingerprint = snapshot_fingerprint(source_ids)
        profile_id = content_hash(conversation_id + fingerprint + json.dumps(data, sort_keys=True))
        with Session(self.engine) as session, session.begin():
            session.execute(
                update(CandidateProfile)
                .where(CandidateProfile.conversation_id == conversation_id)
                .values(is_current=False)
            )
            existing = session.get(CandidateProfile, profile_id)
            if existing is None:
                session.add(
                    CandidateProfile(
                        profile_id=profile_id,
                        conversation_id=conversation_id,
                        request_id=request_id,
                        input_fingerprint=fingerprint,
                        source_ids=source_ids,
                        data=data,
                        status=data["status"],
                        is_current=True,
                        created_at=utcnow(),
                    )
                )
            else:
                existing.is_current = True
        return profile_id

    def profile(self, profile_id, conversation_id):
        with Session(self.engine) as session:
            row = session.get(CandidateProfile, profile_id) if profile_id else None
            if row is None or row.conversation_id != conversation_id:
                return None
            return {
                "profile_id": row.profile_id,
                "data": row.data,
                "status": row.status,
                "input_fingerprint": row.input_fingerprint,
            }

    @staticmethod
    def summary(source):
        result = {
            key: source.get(key)
            for key in (
                "source_id",
                "source_kind",
                "filename",
                "media_type",
                "source_url",
                "content_sha256",
                "pipeline_version",
                "byte_count",
                "status",
                "error_code",
                "error_message",
                "retryable",
                "attempt_count",
                "request_id",
            )
        }
        for key in ("created_at", "processed_at"):
            value = source.get(key)
            result[key] = (
                value.astimezone(UTC).isoformat().replace("+00:00", "Z") if value else None
            )
        return result
