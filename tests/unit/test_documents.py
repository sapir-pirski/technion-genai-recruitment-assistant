import json
import subprocess
from io import BytesIO

import pytest
from docx import Document
from reportlab.pdfgen import canvas

from app.guardrails import (
    GuardrailViolation,
    model_data,
)
from app.ingestion.description import description_html
from app.profiles import (
    MAX_FILE_BYTES,
    ProfileReadError,
    document_converter,
    grounded_sections,
    linkedin_reference,
    merge_profile,
    read_document,
)
from app.schemas import ProfileSections


def test_linkedin_references_are_normalized_and_reject_arbitrary_urls():
    assert (
        linkedin_reference("https://linkedin.com/in/alex-example?trk=tracking")
        == "https://www.linkedin.com/in/alex-example/"
    )
    for value in (
        "https://linkedin.com.evil.test/in/alex",
        "http://linkedin.com/in/alex",
        "https://www.linkedin.com/jobs/1",
        "https://user:secret@linkedin.com/in/alex",
    ):
        with pytest.raises(ProfileReadError):
            linkedin_reference(value)


def test_file_readers_preserve_pdf_docx_table_and_unicode_text():
    stream = BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.drawString(50, 800, "Skills: Python, SQL.")
    pdf.save()
    assert "Python, SQL" in read_document(stream.getvalue(), "profile.pdf")
    doc = Document()
    doc.add_paragraph("Alex Example")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "Python and SQL"
    stream = BytesIO()
    doc.save(stream)
    assert "Python and SQL" in read_document(stream.getvalue(), "profile.docx")
    assert "Résumé" in read_document("Résumé: Python".encode("utf-8-sig"), "profile.txt")
    assert "Python" in read_document("Python".encode("utf-16"), "profile.txt")


def test_invalid_and_oversized_files_fail_before_profile_extraction():
    with pytest.raises(ProfileReadError, match="5 MB"):
        read_document(b"a" * (MAX_FILE_BYTES + 1), "profile.txt")
    with pytest.raises(ProfileReadError, match="24,000"):
        read_document(b"a" * 24001, "profile.txt")
    with pytest.raises(ProfileReadError, match="valid PDF"):
        read_document(b"not PDF", "profile.pdf")
    with pytest.raises(ProfileReadError, match="DOCX"):
        read_document(b"not docx", "profile.docx")
    with pytest.raises(ProfileReadError, match="UTF-8"):
        read_document(b"\xff\x00\xfe\xff", "profile.txt")
    stream = BytesIO()
    pdf = canvas.Canvas(stream)
    pdf.showPage()
    pdf.save()
    with pytest.raises(ProfileReadError, match="OCR"):
        read_document(stream.getvalue(), "profile.pdf")


def test_grounded_merge_keeps_evidence_conflicts_and_unknown_python_years():
    text = "Alex Example. Developer since 2010. Python, SQL."
    sections = ProfileSections(
        name="Alex Example",
        experience=["Developer since 2010"],
        skills=["Python", "SQL"],
        education=[],
        projects=[],
    )
    facts = grounded_sections(sections, text)
    with pytest.raises(ProfileReadError, match="could not be matched"):
        grounded_sections(sections.model_copy(update={"skills": ["Kubernetes"]}), text)
    profile = merge_profile(
        [
            {"source_id": "one", "facts": facts},
            {"source_id": "two", "facts": {**facts, "name": "Different Name"}},
        ]
    )
    assert profile["sections"]["skills"] == ["Python", "SQL"]
    assert len([e for e in profile["evidence"] if e["quote"] == "Python"]) == 2
    assert profile["conflicts"][0]["field"] == "identity"
    assert "Explicit years of Python experience" in profile["missing_information"]
    same_name = merge_profile(
        [
            {"source_id": "one", "facts": {**facts, "name": "Alex Example."}},
            {"source_id": "two", "facts": facts},
        ]
    )
    assert len(same_name["sections"]["name"]) == 1 and not same_name["conflicts"]
    assert {row["quote"] for row in same_name["evidence"] if row["field"] == "name"} == {
        "Alex Example.",
        "Alex Example",
    }


def test_legacy_doc_is_actually_converted_and_read(tmp_path):
    converter = document_converter()
    if not converter:
        pytest.skip("LibreOffice is required for the legacy DOC round trip")
    doc = Document()
    doc.add_paragraph("Alex Example uses Python and SQL.")
    source = tmp_path / "candidate.docx"
    doc.save(source)
    subprocess.run(
        [
            converter,
            "-env:UserInstallation=" + (tmp_path / "office").as_uri(),
            "--headless",
            "--convert-to",
            "doc:MS Word 97",
            "--outdir",
            str(tmp_path),
            str(source),
        ],
        check=True,
        timeout=30,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    legacy = tmp_path / "candidate.doc"
    assert legacy.is_file()
    assert "Python and SQL" in read_document(legacy.read_bytes(), "candidate.doc")


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
        {"history": history, "candidate_profile": {"skills": ["Python"]}}, operation="main"
    )
    assert context["history"][-1] == history[-1]
    assert len(json.dumps(context["history"])) <= 16000
    assert len(history) == 40
    assert context["candidate_profile"]["skills"] == ["Python"]
    with pytest.raises(GuardrailViolation, match="too large"):
        model_data({"candidate_profile": "x" * 48001}, operation="main")
    with pytest.raises(GuardrailViolation, match="credentials"):
        model_data({"answer": SECRET}, operation="judge")
