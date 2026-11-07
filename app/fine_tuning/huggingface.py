"""Explicit local ModernBERT training; benchmark text is never uploaded."""

import hashlib
import json
import os
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.config import ROOT
from app.evaluation.dataset import build_examples, dataset_fingerprint, load_conversations
from app.evaluation.metrics import summarize
from app.fine_tuning.exit_data import INPUT_FORMAT, INPUT_SHA256, LABELS, history_text

BASE_MODEL = "answerdotai/ModernBERT-base"
BASE_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    path.chmod(0o600)


def train(settings, output: Path, *, epochs=5, seed=42, max_length=2048) -> dict:
    # All download/cache writes stay inside the ignored project artifacts directory.
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o700)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    token = settings.hf_token.get_secret_value() or None
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, token=token, trust_remote_code=False
    )
    tokenizer.truncation_side = "left"
    conversations = load_conversations(settings)
    examples = build_examples(conversations)
    split_ids = {
        name: {row["conversation_id"] for row in examples if row["split"] == name}
        for name in ("train", "validation", "test")
    }
    if (
        not all(split_ids.values())
        or split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("Training requires disjoint stored conversation splits.")
    partitions = {
        name: [row for row in examples if row["split"] == name] for name in ("train", "validation")
    }
    texts = {
        name: [history_text(row["history"]) for row in rows] for name, rows in partitions.items()
    }
    encoded = {
        name: tokenizer(
            rows, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        )
        for name, rows in texts.items()
    }
    targets = {
        name: torch.tensor([int(row["label"] == "end") for row in rows])
        for name, rows in partitions.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        revision=BASE_REVISION,
        token=token,
        trust_remote_code=False,
        num_labels=2,
        id2label=dict(enumerate(LABELS)),
        label2id={label: index for index, label in enumerate(LABELS)},
        attn_implementation="eager",
        reference_compile=False,
        classifier_dropout=0.1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    # Class weights derive only from the training partition.
    counts = torch.bincount(targets["train"], minlength=2)
    if not bool((counts > 0).all()):
        raise ValueError("Training must contain both Exit labels.")
    weights = (len(targets["train"]) / (2 * counts.float())).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    batch_size, best_score, best_epoch = 4, (-1.0, -float("inf")), None
    history, started = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        indices = torch.randperm(len(targets["train"]))
        for batch in indices.split(batch_size):
            inputs = {key: value[batch].to(device) for key, value in encoded["train"].items()}
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits
            loss = loss_fn(logits, targets["train"][batch].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch)
        model.eval()
        predictions, validation_loss = [], 0.0
        with torch.inference_mode():
            for batch in torch.arange(len(targets["validation"])).split(batch_size):
                inputs = {
                    key: value[batch].to(device) for key, value in encoded["validation"].items()
                }
                logits = model(**inputs).logits
                validation_loss += torch.nn.functional.cross_entropy(
                    logits, targets["validation"][batch].to(device), reduction="sum"
                ).item()
                predictions.extend(logits.argmax(-1).cpu().tolist())
        rows = [
            {"expected": LABELS[target], "predicted": LABELS[predicted], "error": None}
            for target, predicted in zip(targets["validation"].tolist(), predictions, strict=True)
        ]
        metrics = summarize(rows, list(LABELS))
        validation_loss /= len(rows)
        score = (metrics["classification_report"]["macro avg"]["f1-score"], -validation_loss)
        if score > best_score:
            best_score, best_epoch = score, epoch
            model.save_pretrained(output / "pytorch", safe_serialization=True)
            tokenizer.save_pretrained(output / "pytorch")
        record = {
            "epoch": epoch,
            "training_loss": total_loss / len(targets["train"]),
            "validation_loss": validation_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": score[0],
            "selected": epoch == best_epoch,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "provider": "huggingface",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "created_at": datetime.now(UTC).isoformat(),
        "device": device,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 2e-5,
        "max_length": max_length,
        "input_format": INPUT_FORMAT,
        "input_sha256": INPUT_SHA256,
        "labels": list(LABELS),
        "dataset_version": settings.dataset_version,
        "dataset_sha256": dataset_fingerprint(conversations),
        "split_conversations": {name: len(ids) for name, ids in split_ids.items()},
        "training_examples": len(partitions["train"]),
        "validation_examples": len(partitions["validation"]),
        "test_used_for_selection": False,
        "selected_epoch": best_epoch,
        "selection_metric": "validation macro F1; validation loss breaks ties",
        "history": history,
        "seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output / "training-report.json", report)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    return report


def export_onnx(settings, output: Path) -> dict:
    """Export and quantize locally; runtime needs no PyTorch or Transformers."""
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        output / "pytorch", attn_implementation="eager", reference_compile=False
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(output / "pytorch")
    tokenizer.truncation_side = "left"
    destination = output / "runtime"
    destination.mkdir(exist_ok=True)
    tokenizer.save_pretrained(destination)
    encoded = tokenizer(
        history_text([{"role": "user", "content": "Thanks for the details."}]), return_tensors="pt"
    )

    class Wrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, input_ids, attention_mask):
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.onnx.export(
        Wrapper(model),
        (encoded["input_ids"], encoded["attention_mask"]),
        output / "model-float.onnx",
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quantize_dynamic(
        output / "model-float.onnx",
        destination / "model.onnx",
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(destination / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    )
    # Check export fidelity on validation only. The test partition stays untouched.
    validation = [
        row for row in build_examples(load_conversations(settings)) if row["split"] == "validation"
    ]
    agreement, max_difference = 0, 0.0
    for row in validation:
        inputs = tokenizer(
            history_text(row["history"]), truncation=True, max_length=2048, return_tensors="pt"
        )
        with torch.inference_mode():
            reference = model(**inputs).logits.numpy()
        logits = session.run(
            None,
            {
                key: value.numpy()
                for key, value in inputs.items()
                if key in {"input_ids", "attention_mask"}
            },
        )[0]
        agreement += int(reference.argmax(-1)[0] == logits.argmax(-1)[0])
        max_difference = max(max_difference, float(np.max(np.abs(reference - logits))))
    if agreement != len(validation):
        raise ValueError(
            "Quantized classifier differs from the selected model on validation labels."
        )
    report = json.loads((output / "training-report.json").read_text())
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.iterdir()
        if path.is_file()
    }
    manifest = {
        "format_version": 1,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "input_format": INPUT_FORMAT,
        "input_sha256": INPUT_SHA256,
        "labels": list(LABELS),
        "max_length": report["max_length"],
        "selected_epoch": report["selected_epoch"],
        "dataset_sha256": report["dataset_sha256"],
        "quantization": "dynamic-int8-per-channel; embeddings retained as float32",
        "validation_export_agreement": agreement,
        "validation_examples": len(validation),
        "max_validation_logit_difference": max_difference,
        "files": files,
    }
    write_json(destination / "exit_classifier.json", manifest)
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    return manifest


def activate_local(settings) -> dict:
    from app.agents.backends.huggingface import _load_local
    from app.config import LOCAL_EXIT

    manifest = json.loads((LOCAL_EXIT / "exit_classifier.json").read_text())
    revision = manifest["files"]["model.onnx"]
    _load_local(str(LOCAL_EXIT.resolve()), revision)
    result = {
        "provider": "huggingface",
        "model": manifest["base_model"],
        "artifact_sha256": revision,
        "activated_at": datetime.now(UTC).isoformat(),
        "scope": "this machine only; no uploads or cloud credential changes",
    }
    write_json(LOCAL_EXIT.parent / "active.json", result)
    return result
