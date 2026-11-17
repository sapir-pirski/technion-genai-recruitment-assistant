import hashlib
import json
from collections import Counter
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from app.config import Settings
from app.evaluation.dataset import (
    SPLIT_NAMES,
    dataset_fingerprint,
    load_conversations,
    load_examples,
    prepare_dataset,
    split_examples,
)
from app.evaluation.runner import evaluate, evaluate_local_exit


def file_settings(path, version="test-v1", **changes):
    return Settings(
        _env_file=None,
        database_url="",
        openai_api_key="",
        dataset_path=path,
        dataset_version=version,
        **changes,
    )


def test_json_roundtrip_preserves_splits_and_fingerprint_without_database_or_api(
    benchmark_dataset,
    benchmark_conversations,
    tmp_path,
):
    settings = file_settings(benchmark_dataset)
    original = benchmark_dataset.read_bytes()
    with patch("app.storage.database.create_engine", side_effect=AssertionError("No database")):
        assert load_conversations(settings) == benchmark_conversations
        examples = load_examples(settings)
        manifest = prepare_dataset(settings, tmp_path / "processed")
    assert Counter(row["label"] for row in examples) == {"continue": 3, "schedule": 3, "end": 3}
    assert manifest["source"] == "json" and manifest["dataset_version"] == "test-v1"
    assert manifest["input_sha256"] == hashlib.sha256(original).hexdigest()
    assert manifest["source_sha256"] == benchmark_conversations[0]["source_sha256"]
    assert manifest["dataset_sha256"] == dataset_fingerprint(benchmark_conversations)
    assert manifest["splits"] == {"train": [1], "validation": [2], "test": [3]}
    for split in SPLIT_NAMES:
        path = tmp_path / "processed" / f"{split}.json"
        assert json.loads(path.read_text()) == split_examples(examples, split)
        assert path.stat().st_mode & 0o777 == 0o600
    benchmark_dataset.write_text(json.dumps(benchmark_conversations, indent=4))
    reformatted = prepare_dataset(settings, tmp_path / "reformatted")
    assert reformatted["dataset_sha256"] == manifest["dataset_sha256"]
    assert reformatted["input_sha256"] != manifest["input_sha256"]


def test_original_assignment_json_gets_the_frozen_conversation_split(
    tmp_path, benchmark_conversations
):
    records = []
    for cid in range(1, 16):
        record = deepcopy(benchmark_conversations[(cid - 1) % 3])
        record["conversation_id"] = cid
        for key in ("dataset_version", "source_sha256", "split"):
            record.pop(key)
        records.append(record)
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps(list(reversed(records))))
    settings = file_settings(path, "assignment-v1")
    conversations = load_conversations(settings)
    assert [row["conversation_id"] for row in conversations] == list(range(1, 16))
    assert {
        name: [row["conversation_id"] for row in conversations if row["split"] == name]
        for name in SPLIT_NAMES
    } == {"train": [1, 2, 3, 4, 5, 6, 8, 10, 14], "validation": [7, 11, 13], "test": [9, 12, 15]}
    assert {row["source_sha256"] for row in conversations} == {
        hashlib.sha256(path.read_bytes()).hexdigest()
    }
    conversations[8]["split"] = "train"
    path.write_text(json.dumps(conversations))
    with pytest.raises(ValueError, match="original conversation splits"):
        load_conversations(settings)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("duplicate", "Duplicate conversation"),
        ("chronology", "timestamps are not ordered"),
        ("naive_time", "include a timezone"),
        ("candidate_label", "Invalid candidate turn"),
        ("missing_label", "Missing or invalid recruiter label"),
        ("unknown_label", "Invalid benchmark record"),
        ("missing_split", "train, validation, and test"),
        ("version", "version does not match"),
        ("source_hash", "fingerprints disagree"),
        ("malformed_time", "ISO 8601"),
    ],
)
def test_invalid_json_benchmarks_fail_before_evaluation(
    problem,
    match,
    benchmark_dataset,
    benchmark_conversations,
):
    records = deepcopy(benchmark_conversations)
    if problem == "duplicate":
        records.append(deepcopy(records[0]))
    elif problem == "chronology":
        records[0]["turns"][1]["timestamp_utc"] = "2020-01-01T00:00:00Z"
    elif problem == "naive_time":
        records[0]["start_time_utc"] = "2026-10-01T08:00:00"
    elif problem == "candidate_label":
        records[0]["turns"][1]["label"] = "end"
    elif problem == "missing_label":
        records[0]["turns"][0]["label"] = None
    elif problem == "unknown_label":
        records[0]["turns"][0]["label"] = "private-invalid-label"
    elif problem == "missing_split":
        records.pop()
    elif problem == "version":
        records[0]["dataset_version"] = "another-version"
    elif problem == "source_hash":
        records[0]["source_sha256"] = "0" * 64
    else:
        records[0]["start_time_utc"] = "private-invalid-timestamp"
    benchmark_dataset.write_text(json.dumps(records))
    with pytest.raises(ValueError, match=match) as error:
        load_conversations(file_settings(benchmark_dataset))
    assert "private-invalid" not in str(error.value)


@pytest.mark.parametrize(
    "content", [b"[]", b"{}", b'["private-record"]', b'{"private-text":', b"\xff"]
)
def test_invalid_file_shape_and_encoding_do_not_expose_contents(tmp_path, content):
    path = tmp_path / "benchmark.json"
    path.write_bytes(content)
    with pytest.raises(ValueError) as error:
        load_conversations(file_settings(path))
    assert "private" not in str(error.value)


def test_missing_json_gives_a_local_file_instruction(tmp_path):
    with pytest.raises(ValueError, match="--dataset / DATASET_PATH"):
        load_conversations(file_settings(tmp_path / "missing.json"))


def test_training_checks_the_json_before_downloading_or_creating_model_artifacts(tmp_path):
    from app.fine_tuning.huggingface import train

    output = tmp_path / "model-artifacts"
    with pytest.raises(ValueError, match="Benchmark JSON is missing"):
        train(file_settings(tmp_path / "missing.json"), output)
    assert not output.exists()


def test_missing_database_configuration_is_checked_only_for_runtime_storage(tmp_path):
    from app.storage.database import make_engine

    settings = file_settings(tmp_path / "benchmark.json")
    assert settings.database_url == ""
    with pytest.raises(ValueError, match="DATABASE_URL"):
        make_engine(settings.database_url)
    with pytest.raises(ValueError, match="DATABASE_URL"):
        _ = settings.postgres_dsn


def test_prepare_data_cli_accepts_a_json_path_without_credentials(
    benchmark_dataset, tmp_path, monkeypatch, capsys
):
    from app.main import main

    monkeypatch.setattr(
        "sys.argv",
        [
            "recruitment",
            "prepare-data",
            "--dataset",
            str(benchmark_dataset),
            "--dataset-version",
            "test-v1",
            "--output",
            str(tmp_path / "prepared"),
        ],
    )
    monkeypatch.setattr(
        "app.main.Settings",
        lambda **kwargs: Settings(
            _env_file=None,
            database_url="",
            openai_api_key="",
            **kwargs,
        ),
    )
    main()
    assert json.loads(capsys.readouterr().out)["source"] == "json"


def test_exit_evaluations_use_json_without_initializing_the_application(
    benchmark_dataset, tmp_path
):
    settings = file_settings(benchmark_dataset)
    backend = Mock(exit_label=Mock(return_value="do_not_end"))
    with (
        patch("app.evaluation.runner.make_engine", side_effect=AssertionError("No database")),
        patch("app.evaluation.runner.ChatService", side_effect=AssertionError("No application")),
        patch("app.evaluation.runner.OpenAIBackend", return_value=backend),
    ):
        report = evaluate(
            settings.model_copy(update={"openai_api_key": SecretStr("test-only-key")}), task="exit"
        )
    assert report["all_turns"]["count"] == 3 and report["all_turns"]["failures"] == 0
    assert report["calendar_context"] == "not used by Exit classification"
    conversations = load_conversations(settings)
    (tmp_path / "exit_classifier.json").write_text(
        json.dumps(
            {
                "dataset_sha256": dataset_fingerprint(conversations),
                "base_model": "test-classifier",
                "selected_epoch": 1,
            }
        )
    )
    (tmp_path / "model.onnx").write_bytes(b"test-only-placeholder")
    with (
        patch("app.evaluation.runner.make_engine", side_effect=AssertionError("No database")),
        patch("app.agents.backends.huggingface.exit_label", return_value="do_not_end"),
    ):
        local_report = evaluate_local_exit(settings, tmp_path)
    assert local_report["all_turns"]["count"] == 3 and local_report["all_turns"]["failures"] == 0
