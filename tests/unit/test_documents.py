import json

import pytest

from app.guardrails import GuardrailViolation, model_data
from app.rag.description import description_html

SECRET = "sk-proj-" + "FICTITIOUS_TEST_CREDENTIAL_" * 2


def test_pdf_description_markup_cannot_inject_scripts_images_or_unsafe_links():
    rendered = description_html(
        "## Responsibilities\n\n- Build **Python APIs**.\n\n"
        "<script>alert(1)</script>\n\n![image](https://example.com/pixel)\n\n"
        "[Bad link](javascript:alert(1))"
    )
    assert "<h2>Responsibilities</h2>" in rendered
    assert "<strong>Python APIs</strong>" in rendered
    assert "<script>" not in rendered and "<img" not in rendered
    assert 'href="javascript:' not in rendered


def test_context_budget_preserves_latest_message_and_full_stored_history():
    history = [{"role": "user", "content": str(index) + "x" * 3000} for index in range(40)]
    context = model_data(
        {"history": history, "job": {"title": "Python Developer"}}, operation="main"
    )
    assert context["history"][-1] == history[-1]
    assert len(json.dumps(context["history"])) <= 16000
    assert len(history) == 40
    assert context["job"]["title"] == "Python Developer"
    with pytest.raises(GuardrailViolation, match="too large"):
        model_data({"job": "x" * 48001}, operation="main")
    with pytest.raises(GuardrailViolation, match="credentials"):
        model_data({"answer": SECRET}, operation="judge")
