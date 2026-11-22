"""Local ModernBERT classifier; CPU inference without PyTorch or network calls."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.fine_tuning.exit_data import INPUT_SHA256, LABELS, classifier_spec, history_text
from app.observability.tracing import span


@lru_cache(maxsize=2)
def _load_local(directory, revision):
    artifacts = _load_files(lambda filename: str(Path(directory) / filename))
    if artifacts[2]["files"]["model.onnx"] != revision:
        raise ValueError("Exit classifier differs from the configured artifact revision.")
    return artifacts


def _load_files(resolve):
    import onnxruntime as ort
    from tokenizers import Tokenizer

    paths, manifest = _validated_files(resolve)
    tokenizer = Tokenizer.from_file(paths["tokenizer.json"])
    tokenizer.enable_truncation(max_length=manifest["max_length"], direction="left")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        paths["model.onnx"], sess_options=options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, session, manifest


def verify_local_artifact(directory, revision):
    _, manifest = _validated_files(lambda filename: str(Path(directory) / filename))
    if manifest["files"]["model.onnx"] != revision:
        raise ValueError("Exit classifier differs from the configured artifact revision.")
    return manifest


def _validated_files(resolve):
    with open(resolve("exit_classifier.json")) as handle:
        manifest = json.load(handle)
    if (
        manifest.get("format_version") != 1
        or manifest.get("input_sha256") != INPUT_SHA256
        or manifest.get("labels") != list(LABELS)
        or not isinstance(manifest.get("max_length"), int)
        or not 1 <= manifest["max_length"] <= 8192
    ):
        raise ValueError("Exit classifier artifact does not match the runtime input contract.")
    paths = {}
    for filename in ("tokenizer.json", "model.onnx"):
        path = resolve(filename)
        with open(path, "rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != manifest.get("files", {}).get(filename):
            raise ValueError("Exit classifier artifact checksum mismatch.")
        paths[filename] = path
    return paths, manifest


def exit_label(settings, history, *, artifact_path=None):
    repository, revision = settings.hf_exit_model, settings.hf_exit_revision
    with span(
        "classifier",
        "exit",
        model=repository,
        attributes={
            **classifier_spec(repository, revision),
            "provider": "huggingface",
            "runtime": "onnxruntime-cpu",
            "usage_source": "local_tokenizer",
            "usage_reported": False,
            "billable_tokens": False,
            "estimated_cost_usd": 0.0,
            "price_version": "local-inference-no-api-charge",
            "artifact_source": "local",
        },
    ) as item:
        if not history:
            item.input_tokens = item.output_tokens = 0
            item.attributes["decision"] = "do_not_end"
            item.attributes["empty_history"] = True
            return "do_not_end"
        directory = Path(artifact_path or settings.hf_exit_local_path).resolve()
        tokenizer, session, manifest = _load_local(str(directory), revision)
        encoding = tokenizer.encode(history_text(history))
        item.input_tokens, item.output_tokens = len(encoding.ids), 0
        item.attributes["input_truncated"] = bool(encoding.overflowing)
        inputs = {
            "input_ids": np.asarray([encoding.ids], dtype=np.int64),
            "attention_mask": np.asarray([encoding.attention_mask], dtype=np.int64),
        }
        logits = session.run(["logits"], inputs)[0]
        if logits.shape != (1, 2) or not np.isfinite(logits).all():
            raise ValueError("Exit classifier returned invalid logits.")
        label = manifest["labels"][int(logits.argmax(axis=-1)[0])]
        item.attributes["decision"] = label
        return label
