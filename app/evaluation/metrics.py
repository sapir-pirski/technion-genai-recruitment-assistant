from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

LABELS = ["continue", "schedule", "end"]


def summarize(rows: list[dict], labels=None) -> dict:
    labels = labels or LABELS
    if not rows:
        return {
            "count": 0,
            "accuracy": None,
            "successful_call_accuracy": None,
            "failures": 0,
            "failure_rate": None,
            "coverage": None,
            "labels": labels,
            "confusion_matrix": [],
            "classification_report": {},
        }
    gold = [row["expected"] for row in rows]
    failed = [bool(row.get("error")) or row.get("predicted") not in labels for row in rows]
    predicted = [
        "failure" if bad else row["predicted"] for row, bad in zip(rows, failed, strict=True)
    ]
    successful = [row for row, bad in zip(rows, failed, strict=True) if not bad]
    labels = [*labels, "failure"] if any(failed) else labels
    return {
        "count": len(rows),
        "accuracy": accuracy_score(gold, predicted),
        "labels": labels,
        "confusion_matrix": confusion_matrix(gold, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            gold, predicted, labels=labels, output_dict=True, zero_division=0
        ),
        "failures": sum(failed),
        "failure_rate": sum(failed) / len(rows),
        "coverage": len(successful) / len(rows),
        "successful_call_accuracy": accuracy_score(
            [row["expected"] for row in successful], [row["predicted"] for row in successful]
        )
        if successful
        else None,
    }
