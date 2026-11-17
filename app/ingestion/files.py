"""The assignment PDF is the application's only job-description input."""

from pathlib import Path

from app.config import ROOT

ORIGINAL_PDF_NAME = "Python Developer Job Description.pdf"


def resolve_pdf(settings, value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    expected = settings.job_pdf_path.resolve() / ORIGINAL_PDF_NAME
    if path != expected:
        raise ValueError(f"Only data/jobs/{ORIGINAL_PDF_NAME} can be ingested")
    return path
