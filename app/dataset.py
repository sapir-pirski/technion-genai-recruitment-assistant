import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings

SPLIT_NAMES = ("train", "validation", "test")
# Keep the original conversation split fixed across preparation, training and evaluation.
ASSIGNMENT_SPLITS = {
    "train": (1, 2, 3, 4, 5, 6, 8, 10, 14),
    "validation": (7, 11, 13),
    "test": (9, 12, 15),
}


class BenchmarkTurn(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
    turn_id: int = Field(gt=0)
    speaker: Literal["candidate", "recruiter"]
    timestamp_utc: str
    text: str = Field(min_length=1)
    label: Literal["continue", "schedule", "end"] | None


class BenchmarkConversation(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", hide_input_in_errors=True)
    conversation_id: int = Field(gt=0)
    start_time_utc: str
    candidate_phone: str | int | None = None
    recruiter_phone: str | int | None = None
    dataset_version: str
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split: Literal["train", "validation", "test"]
    turns: list[BenchmarkTurn] = Field(min_length=1)


def redact(text: str) -> str:
    text = re.sub(
        r"\+?\d[\d ()-]{7,}\d",
        lambda match: "[PHONE]" if len(re.sub(r"\D", "", match.group())) >= 10 else match.group(),
        text,
    )
    return re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)


def load_conversations(settings: Settings) -> list[dict]:
    """Read the local assignment JSON or a normalized snapshot, without SQL access."""
    return _load_snapshot(settings)[0]


def _load_snapshot(settings):
    try:
        raw = settings.dataset_path.read_bytes()
    except FileNotFoundError:
        raise ValueError(
            "Benchmark JSON is missing. Place the authorized dataset at "
            "data/sms_conversations.json or select it with --dataset / DATASET_PATH."
        ) from None
    try:
        records = json.loads(raw.decode("utf-8-sig"))
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Benchmark input must be valid UTF-8 JSON.") from None
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark JSON must contain a nonempty list of conversations.")
    input_hash = hashlib.sha256(raw).hexdigest()
    conversations = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Invalid benchmark record {index}: expected a conversation object.")
        record = dict(record)
        record.setdefault("dataset_version", settings.dataset_version)
        record.setdefault("source_sha256", input_hash)
        if "split" not in record and settings.dataset_version == "assignment-v1":
            record["split"] = next(
                (
                    name
                    for name, ids in ASSIGNMENT_SPLITS.items()
                    if record.get("conversation_id") in ids
                ),
                None,
            )
        try:
            conversation = BenchmarkConversation.model_validate(record).model_dump()
        except ValidationError:
            raise ValueError(
                f"Invalid benchmark record {index}: check IDs, required fields, speakers, labels and split."
            ) from None
        if conversation["dataset_version"] != settings.dataset_version:
            raise ValueError(
                "Benchmark version does not match --dataset-version / DATASET_VERSION."
            )
        if settings.dataset_version == "assignment-v1" and (
            conversation["conversation_id"] not in ASSIGNMENT_SPLITS[conversation["split"]]
        ):
            raise ValueError(
                "The assignment benchmark must retain its original conversation splits."
            )
        conversations.append(conversation)
    conversations.sort(key=lambda row: row["conversation_id"])
    examples = build_examples(conversations)
    if settings.dataset_version == "assignment-v1" and (
        {row["conversation_id"] for row in conversations}
        != {cid for ids in ASSIGNMENT_SPLITS.values() for cid in ids}
    ):
        raise ValueError("The assignment benchmark must include all 15 original conversations.")
    if {row["conversation_id"] for row in examples} != {
        row["conversation_id"] for row in conversations
    }:
        raise ValueError("Every benchmark conversation must contain a labeled recruiter turn.")
    if {row["split"] for row in conversations} != set(SPLIT_NAMES):
        raise ValueError("The benchmark must contain train, validation, and test conversations.")
    if len({row["source_sha256"] for row in conversations}) != 1:
        raise ValueError("Dataset source fingerprints disagree within one version.")
    return conversations, input_hash


def _timestamp(value):
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("Dataset timestamps must use ISO 8601 with a timezone.") from None
    if instant.tzinfo is None:
        raise ValueError("Dataset timestamps must include a timezone.")
    return instant


def build_examples(conversations: list[dict]) -> list[dict]:
    seen = set()
    examples = []
    for conversation in conversations:
        cid = conversation["conversation_id"]
        if cid in seen:
            raise ValueError("Duplicate conversation ID.")
        seen.add(cid)
        if conversation["split"] not in SPLIT_NAMES:
            raise ValueError("Each conversation needs a train, validation, or test split.")
        history = []
        previous_time = _timestamp(conversation["start_time_utc"])
        previous_turn_id = 0
        for turn in conversation["turns"]:
            instant = _timestamp(turn["timestamp_utc"])
            if instant < previous_time:
                raise ValueError("Conversation timestamps are not ordered.")
            if turn["turn_id"] <= previous_turn_id:
                raise ValueError("Conversation turn IDs are not ordered.")
            if turn["speaker"] == "recruiter":
                if turn["label"] not in ("continue", "schedule", "end"):
                    raise ValueError("Missing or invalid recruiter label.")
                examples.append(
                    {
                        "conversation_id": cid,
                        "turn_id": turn["turn_id"],
                        "history": [dict(message) for message in history],
                        "reference_time": previous_time.isoformat(),
                        "label": turn["label"],
                        "is_opening": not history,
                        "split": conversation["split"],
                    }
                )
            elif turn["speaker"] != "candidate" or turn["label"] is not None:
                raise ValueError("Invalid candidate turn.")
            history.append(
                {
                    "role": "assistant" if turn["speaker"] == "recruiter" else "user",
                    "content": redact(turn["text"]),
                }
            )
            previous_time = instant
            previous_turn_id = turn["turn_id"]
    return examples


def load_examples(settings: Settings) -> list[dict]:
    return build_examples(load_conversations(settings))


def dataset_fingerprint(conversations: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(conversations, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def split_examples(examples: list[dict], split: str) -> list[dict]:
    if split == "all":
        return examples
    if split not in SPLIT_NAMES:
        raise ValueError("Choose train, validation, test, or all.")
    return [item for item in examples if item["split"] == split]


def prepare_dataset(settings: Settings, output: Path) -> dict:
    conversations, input_hash = _load_snapshot(settings)
    examples = build_examples(conversations)
    splits = {
        name: [row["conversation_id"] for row in conversations if row["split"] == name]
        for name in SPLIT_NAMES
    }
    source_hashes = {row["source_sha256"] for row in conversations}
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    manifest = {
        "source": "json",
        "input_sha256": input_hash,
        "dataset_version": settings.dataset_version,
        "source_sha256": source_hashes.pop(),
        "dataset_sha256": dataset_fingerprint(conversations),
        "splits": splits,
        "rationale": "Each conversation belongs to exactly one frozen train/validation/test split.",
        "conversations": len(conversations),
        "turns": sum(len(row["turns"]) for row in conversations),
        "examples": len(examples),
        "labels": dict(Counter(x["label"] for x in examples)),
    }
    for split in SPLIT_NAMES:
        rows = split_examples(examples, split)
        (output / f"{split}.json").write_text(json.dumps(rows, indent=2))
        (output / f"{split}.json").chmod(0o600)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.json").chmod(0o600)
    return manifest
