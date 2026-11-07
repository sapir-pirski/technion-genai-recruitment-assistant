import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.config import ROOT
from app.ingestion.files import resolve_pdf
from app.ingestion.normalize import digest


def export_corpus(settings, destination):
    """Export the single PDF folder and its metadata, never credentials or DBs."""
    manifest = json.loads(settings.job_manifest_path.read_text())
    files = {"config/job_corpus_manifest.json": settings.job_manifest_path}
    if settings.job_sources_path.exists():
        files["config/job_sources.json"] = settings.job_sources_path
    for record in manifest["jobs"]:
        path = resolve_pdf(settings, record["pdf_path"])
        if digest(path.read_bytes()) != record["pdf_sha256"]:
            raise ValueError("A PDF changed since ingestion; run ingest before exporting")
        try:
            relative = path.relative_to(ROOT)
        except ValueError as exc:
            raise ValueError("Export only supports corpus files inside the repository") from exc
        files[str(relative)] = path
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.zip")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for name, source in sorted(files.items()):
            archive.write(source, name)
        archive.writestr(
            "CORPUS_README.txt",
            "Job descriptions from data/jobs/ and their centralized manifest.\nExtract at the repository root, configure OPENAI_API_KEY, then run:\npython -m app.main ingest\nEmbedding API calls incur usage charges. Job descriptions retain their owners' rights. This archive does not include credentials, databases, candidate conversations, or historical source snapshots.\n",
        )
    temporary.replace(destination)
    return {"archive": str(destination), "documents": len(manifest["jobs"]), "files": len(files)}
