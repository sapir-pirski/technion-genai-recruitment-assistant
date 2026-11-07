from datetime import UTC

from sqlalchemy import and_, inspect, select
from sqlalchemy.orm import Session

from app.storage.models import DatasetConversation, DatasetTurn


class DatasetRepository:
    """Read verified benchmark data without mixing it with live conversations."""

    def __init__(self, engine, dataset_version):
        self.engine, self.dataset_version = engine, dataset_version

    def load_conversations(self):
        error = (
            f"Benchmark dataset '{self.dataset_version}' is missing from PostgreSQL. "
            "Restore dataset_conversations and dataset_turns from a database backup "
            "before preparing data or evaluating models."
        )
        inspector = inspect(self.engine)
        if not all(
            inspector.has_table(name) for name in ("dataset_conversations", "dataset_turns")
        ):
            raise ValueError(error)
        records = {}
        with Session(self.engine) as session:
            rows = session.execute(
                select(DatasetConversation, DatasetTurn)
                .outerjoin(
                    DatasetTurn,
                    and_(
                        DatasetConversation.dataset_version == DatasetTurn.dataset_version,
                        DatasetConversation.conversation_id == DatasetTurn.conversation_id,
                    ),
                )
                .where(DatasetConversation.dataset_version == self.dataset_version)
                .order_by(DatasetConversation.conversation_id, DatasetTurn.turn_id)
            )
            for conversation, turn in rows:
                record = records.setdefault(
                    conversation.conversation_id,
                    {
                        "conversation_id": conversation.conversation_id,
                        "start_time_utc": conversation.started_at.astimezone(UTC)
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "candidate_phone": conversation.source_metadata.get("candidate_phone"),
                        "recruiter_phone": conversation.source_metadata.get("recruiter_phone"),
                        "dataset_version": conversation.dataset_version,
                        "source_sha256": conversation.source_sha256,
                        "split": conversation.split,
                        "turns": [],
                    },
                )
                if turn is not None:
                    record["turns"].append(
                        {
                            "turn_id": turn.turn_id,
                            "speaker": turn.speaker,
                            "timestamp_utc": turn.occurred_at.astimezone(UTC)
                            .isoformat()
                            .replace("+00:00", "Z"),
                            "text": turn.text,
                            "label": turn.expected_action,
                        }
                    )
        if not records:
            raise ValueError(error)
        if any(not record["turns"] for record in records.values()):
            raise ValueError(
                "The benchmark contains conversations without turns; restore a complete dataset backup"
            )
        return list(records.values())
