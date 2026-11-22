"""Names and boundaries for the single PDF input directory."""

import json
import re
import unicodedata
from pathlib import Path

from app.config import ROOT
from app.ingestion.normalize import digest

ORIGINAL_PDF_NAME = "Python Developer Job Description.pdf"


def resolve_pdf(settings, value):
    path = Path(value)
    path = (path if path.is_absolute() else ROOT / path).resolve()
    if path.parent != settings.job_pdf_path.resolve() or path.suffix.lower() != ".pdf":
        raise ValueError("Place PDFs directly in the jobs folder before ingestion")
    return path


def job_pdf_filename(job, settings):
    # Keep a posting's established name when its description or title changes.
    if settings.job_manifest_path.exists():
        manifest = json.loads(settings.job_manifest_path.read_text())
        for record in manifest["jobs"]:
            if record["job_id"] == job["job_id"] and record["document_id"].startswith("posting-"):
                name = Path(record["pdf_path"]).name
                if re.fullmatch(r"[a-z0-9-]+__[a-z0-9-]+__[a-f0-9]{8}\.pdf", name):
                    return name

    def slug(value, limit):
        value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:limit].rstrip("-")

    identity = job["job_id"].removeprefix("job-")
    short_id = identity[:8] if re.fullmatch(r"[a-f0-9]{8,}", identity) else digest(identity)[:8]
    title = slug(job["title"], 90) or "job-description"
    company = slug(job["company_name"], 60) or "company"
    return f"{title}__{company}__{short_id}.pdf"
