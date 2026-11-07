import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.agents.backends.common import prompt_spec
from app.config import Settings
from app.storage.database import make_engine
from app.storage.dataset_repository import DatasetRepository

SPLIT_NAMES = ("train", "validation", "test")


def redact(text: str) -> str:
    text = re.sub(
        r"\+?\d[\d ()-]{7,}\d",
        lambda match: "[PHONE]" if len(re.sub(r"\D", "", match.group())) >= 10 else match.group(),
        text,
    )
    return re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)


def load_conversations(settings: Settings) -> list[dict]:
    engine = make_engine(settings.database_url)
    try:
        return DatasetRepository(engine, settings.dataset_version).load_conversations()
    finally:
        engine.dispose()


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
        previous_time = datetime.fromisoformat(
            conversation["start_time_utc"].replace("Z", "+00:00")
        )
        if previous_time.tzinfo is None:
            raise ValueError("Dataset timestamps must include a timezone.")
        previous_turn_id = 0
        for turn in conversation["turns"]:
            instant = datetime.fromisoformat(turn["timestamp_utc"].replace("Z", "+00:00"))
            if instant.tzinfo is None:
                raise ValueError("Dataset timestamps must include a timezone.")
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


def prepare_dataset(
    settings: Settings, output: Path, *, exit_base_model: str | None = None
) -> dict:
    base_model = exit_base_model or settings.chat_model
    exit_prompt = prompt_spec("exit", base_model)
    conversations = load_conversations(settings)
    examples = build_examples(conversations)
    actual = {row["conversation_id"] for row in examples}
    if actual != {row["conversation_id"] for row in conversations}:
        raise ValueError("Every benchmark conversation must contain a labeled recruiter turn.")
    splits = {
        name: [row["conversation_id"] for row in conversations if row["split"] == name]
        for name in SPLIT_NAMES
    }
    if not all(splits.values()):
        raise ValueError("The benchmark must contain train, validation, and test conversations.")
    source_hashes = {row["source_sha256"] for row in conversations}
    if len(source_hashes) != 1:
        raise ValueError("Dataset source fingerprints disagree within one version.")
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    manifest = {
        "source": "postgresql",
        "dataset_version": settings.dataset_version,
        "source_sha256": source_hashes.pop(),
        "dataset_sha256": dataset_fingerprint(conversations),
        "splits": splits,
        "rationale": "Each conversation belongs to exactly one stored train/validation/test split.",
        "conversations": len(conversations),
        "turns": sum(len(row["turns"]) for row in conversations),
        "examples": len(examples),
        "labels": dict(Counter(x["label"] for x in examples)),
        "exit_training": {
            "base_model": base_model,
            **{key: value for key, value in exit_prompt.items() if key != "content"},
        },
    }
    for split in SPLIT_NAMES:
        rows = split_examples(examples, split)
        (output / f"{split}.json").write_text(json.dumps(rows, indent=2))
        (output / f"{split}.json").chmod(0o600)
        training = [
            {
                "messages": [
                    {"role": exit_prompt["instruction_role"], "content": exit_prompt["content"]},
                    {
                        "role": "user",
                        "content": json.dumps({"history": row["history"]}, ensure_ascii=False),
                    },
                    {
                        "role": "assistant",
                        "content": "end" if row["label"] == "end" else "do_not_end",
                    },
                ]
            }
            for row in rows
        ]
        (output / f"exit_{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in training) + "\n"
        )
        (output / f"exit_{split}.jsonl").chmod(0o600)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.json").chmod(0o600)
    return manifest
