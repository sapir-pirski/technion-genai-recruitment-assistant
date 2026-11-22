import json
import subprocess
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from docx import Document
from reportlab.pdfgen import canvas

from app.guardrails import (
    GuardrailViolation,
    model_data,
)
from app.ingestion.description import (
    description_html,
    html_markdown,
)
from app.ingestion.logos import fetch_public, normalize_logo
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


@pytest.mark.parametrize(
    "payload",
    [
        b'<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/passwd">]><svg/>',
        b'<svg xmlns="http://www.w3.org/2000/svg"><use href="https://example.com/remote.svg"/></svg>',
    ],
)
def test_untrusted_svg_references_are_rejected(payload):
    with pytest.raises(ValueError):
        normalize_logo(payload)


def test_redirect_to_a_private_host_is_rejected_before_the_request(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))]
    )
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/logo.png"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="public"):
            fetch_public(
                client,
                "https://company.example/logo.png",
                SimpleNamespace(max_retries=0, request_timeout=1),
            )
    assert requests == ["https://company.example/logo.png"]


def test_source_structure_and_job_boundaries_survive_collection():
    payload = """<nav>Other jobs</nav><ul><li><a href="#job"><h5>Python Developer</h5>
    <h5>Location</h5><h5>London</h5></a><h3>Responsibilities</h3><ul>
    <li>Build <strong>Python APIs</strong>.</li><li>Use cloud services:<ul>
    <li>AWS: S3, Lambda</li><li>GCP: BigQuery</li></ul></li></ul>
    <h3>Interview process</h3><ol><li>Intro call</li><li>Technical interview</li></ol>
    <a href="https://example.com/apply">Apply now</a><h3>Next role</h3><p>Unrelated work</p></li></ul>"""
    value = html_markdown(
        payload, title="Python Developer", start_marker="Python Developer", end_marker="Next role"
    )
    rendered = description_html(value, title="Python Developer")
    assert "Other jobs" not in rendered and "Unrelated work" not in rendered
    assert "Apply now" not in rendered
    assert "**Location:** London" in value
    assert "<strong>Python APIs</strong>" in rendered
    assert rendered.count("<ul>") == 2 and "<ol>" in rendered
    assert "  - AWS: S3, Lambda" in value


def test_source_markup_cannot_inject_scripts_images_or_unsafe_links():
    value = html_markdown(
        "<h1>Python Developer</h1><p>Use <b>Python</b> safely.</p>"
        '<script>alert(1)</script><img src="https://example.com/track">'
        '<a href="javascript:alert(1)">Bad link</a>',
        title="Python Developer",
    )
    rendered = description_html(
        value + "\n\n<script>alert(2)</script>\n\n![image](https://example.com/pixel)"
    )
    assert "<script>" not in rendered and "<img" not in rendered
    assert "javascript:" not in rendered


SECRET = "sk-proj-" + "FICTITIOUS_TEST_CREDENTIAL_" * 2


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
