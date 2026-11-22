"""Presentation-only transformations; stored replies and receipts remain complete."""


def response_text(content, result, *, cards=False):
    if not cards:
        return content
    slots = result.get("offered_slots", [])
    if slots:
        block = "Available appointments:\n" + "\n".join(
            f"{index}. {slot['label']}" for index, slot in enumerate(slots, 1)
        )
        if block in content:
            content = content.replace(block, "", 1).replace(
                "Which appointment would you like to confirm?", "", 1
            )
    if booking := result.get("booking"):
        content = content.replace(
            f"Your interview is confirmed for {booking['slot']['label']}.", "", 1
        )
    return "\n\n".join(part.strip() for part in content.split("\n\n") if part.strip())
