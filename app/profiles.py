"""Bounded candidate document readers and source-grounded profile facts."""

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit, urlunsplit
from zipfile import ZipFile

import pymupdf
from docx import Document
from docx.table import Table

from app.schemas import ProfileSections

MAX_PROFILE_CHARS = 24_000
MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_PDF_PAGES = 10
PIPELINE_VERSION = "candidate-context-v2"
FIELDS = ("name", "experience", "skills", "education", "projects")
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
}


class ProfileReadError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def content_hash(value):
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def text_key(value):
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def safe_filename(value):
    name = PurePosixPath((value or "document").replace("\\", "/")).name
    return re.sub(r"[\x00-\x1f\x7f]", "", name)[:200] or "document"


def linkedin_reference(value):
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        valid = (
            parsed.scheme == "https"
            and parsed.hostname in {"linkedin.com", "www.linkedin.com"}
            and not parsed.username
            and not parsed.password
            and parsed.port in {None, 443}
            and re.fullmatch(r"/in/[\w%-]+/?", parsed.path)
        )
    except ValueError:
        valid = False
    if not valid:
        raise ProfileReadError(
            "InvalidLinkedInURL",
            "Use a LinkedIn profile URL beginning https://www.linkedin.com/in/.",
        )
    return urlunsplit(("https", "www.linkedin.com", parsed.path.rstrip("/") + "/", "", ""))


def bounded_text(value):
    value = value.strip()
    if not value:
        raise ProfileReadError(
            "EmptyDocument",
            "No readable text was found. Paste your experience or use a text-searchable document.",
        )
    if len(value) > MAX_PROFILE_CHARS:
        raise ProfileReadError(
            "TextLimitExceeded", "The document exceeds 24,000 characters. Upload a shorter version."
        )
    if any(ord(char) < 32 and char not in "\n\r\t\f" for char in value):
        raise ProfileReadError(
            "InvalidText",
            "The document contains unreadable text. Save it as UTF-8 TXT or use another format.",
        )
    return value


def profile_pdf_text(data):
    if not data.lstrip().startswith(b"%PDF-"):
        raise ProfileReadError("InvalidPDF", "The file is not a valid PDF.")
    try:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ProfileReadError(
                    "PasswordProtected", "Use a document without password protection."
                )
            if not 1 <= len(document) <= MAX_PDF_PAGES:
                raise ProfileReadError("PageLimitExceeded", "Use a PDF with at most 10 pages.")
            parts = []
            for page in document:
                value = page.get_text(sort=True).strip()
                if not value and page.get_images():
                    raise ProfileReadError(
                        "OCRRequired",
                        "The PDF needs OCR. Paste its text or upload a text-searchable version.",
                    )
                parts.append(value)
                if sum(map(len, parts)) > MAX_PROFILE_CHARS:
                    raise ProfileReadError(
                        "TextLimitExceeded",
                        "The document exceeds 24,000 characters. Upload a shorter version.",
                    )
            if not any(parts):
                raise ProfileReadError(
                    "OCRRequired", "The PDF has no readable text. Paste its text or provide OCR."
                )
            return bounded_text("\n\n".join(parts))
    except ProfileReadError:
        raise
    except Exception:
        raise ProfileReadError(
            "InvalidPDF", "The PDF could not be read. Use another document or paste its text."
        ) from None


def docx_text(data):
    try:
        with ZipFile(BytesIO(data)) as archive:
            entries = archive.infolist()
            if len(entries) > 1000 or sum(entry.file_size for entry in entries) > 64 * 1024 * 1024:
                raise ProfileReadError(
                    "ExpandedFileLimit", "The Word document is too large to read safely."
                )
            names = {entry.filename for entry in entries}
            if "word/document.xml" not in names or any(
                "vbaproject" in name.casefold() for name in names
            ):
                raise ProfileReadError(
                    "InvalidDOCX", "Use a standard DOCX document without macros."
                )
        document = Document(BytesIO(data))
        parts = []

        def read_blocks(container):
            for block in container.iter_inner_content():
                if isinstance(block, Table):
                    for row in block.rows:
                        for cell in row.cells:
                            read_blocks(cell)
                else:
                    # Include text boxes as well as ordinary paragraph runs.
                    parts.append("".join(block._p.xpath(".//w:t/text()")))

        read_blocks(document)
        return bounded_text("\n".join(parts))
    except ProfileReadError:
        raise
    except Exception:
        raise ProfileReadError(
            "InvalidDOCX", "The DOCX could not be read. Remove password protection or use PDF/TXT."
        ) from None


def document_converter():
    return (
        shutil.which("soffice")
        or shutil.which("libreoffice")
        or (
            "/Applications/LibreOffice.app/Contents/MacOS/soffice"
            if Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").is_file()
            else None
        )
    )


def legacy_doc_text(data):
    if not data.startswith(bytes.fromhex("d0cf11e0a1b11ae1")):
        raise ProfileReadError(
            "InvalidDOC", "The file is not a legacy Word DOC document. Use its actual file format."
        )
    converter = document_converter()
    if not converter:
        raise ProfileReadError(
            "ConverterUnavailable",
            "DOC reading requires LibreOffice on the server. Upload DOCX, PDF or TXT instead.",
        )
    with tempfile.TemporaryDirectory(prefix="candidate-document-") as directory:
        root = Path(directory)
        profile = root / "office-profile"
        profile.mkdir()
        (profile / "registrymodifications.xcu").write_text(
            '<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry">'
            '<item oor:path="/org.openoffice.Office.Common/Security/Scripting">'
            '<prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item></oor:items>'
        )
        source = root / "profile.doc"
        source.write_bytes(data)
        try:
            subprocess.run(
                [
                    converter,
                    "-env:UserInstallation=" + profile.as_uri(),
                    "--headless",
                    "--norestore",
                    "--convert-to",
                    "docx",
                    "--outdir",
                    str(root),
                    str(source),
                ],
                timeout=30,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            converted = root / "profile.docx"
            if not converted.is_file() or converted.stat().st_size > MAX_FILE_BYTES:
                raise ProfileReadError(
                    "InvalidDOC", "The DOC could not be converted. Upload DOCX, PDF or TXT instead."
                )
            return docx_text(converted.read_bytes())
        except (subprocess.SubprocessError, OSError):
            raise ProfileReadError(
                "DOCConversionFailed",
                "The DOC could not be converted within 30 seconds. Upload DOCX, PDF or TXT instead.",
            ) from None


def read_document(data, filename):
    if not data or len(data) > MAX_FILE_BYTES:
        raise ProfileReadError("FileSizeLimit", "Upload a document no larger than 5 MB.")
    suffix = Path(filename).suffix.casefold()
    if suffix == ".pdf":
        return profile_pdf_text(data)
    if suffix == ".docx":
        return docx_text(data)
    if suffix == ".doc":
        return legacy_doc_text(data)
    if suffix == ".txt":
        if data.lstrip().startswith((b"%PDF-", b"PK\x03\x04", bytes.fromhex("d0cf11e0a1b11ae1"))):
            raise ProfileReadError(
                "FileTypeMismatch", "Use the document's actual PDF or Word file extension."
            )
        encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
        try:
            return bounded_text(data.decode(encoding))
        except UnicodeError:
            raise ProfileReadError(
                "UnsupportedEncoding",
                "Save the TXT document as UTF-8 or UTF-16 and upload it again.",
            ) from None
    raise ProfileReadError("UnsupportedFileType", "Upload a PDF, DOC, DOCX or TXT document.")


def grounded_sections(
    value, source_text, *, allow_terminal_punctuation=False, drop_ungrounded=False
):
    sections = ProfileSections.model_validate(value).model_dump()
    source_key = text_key(source_text)
    for field in FIELDS:
        excerpts = (
            [sections[field]]
            if field == "name" and sections[field]
            else sections[field]
            if field != "name"
            else []
        )
        grounded = []
        for excerpt in excerpts:
            # Chat extraction may add a sentence stop before an original comma.
            # Drop only that terminal punctuation; every retained word must match.
            if allow_terminal_punctuation and text_key(excerpt) not in source_key:
                excerpt = excerpt.rstrip(" \t\r\n.,;:!?")
            if not excerpt.strip() or text_key(excerpt) not in source_key:
                if drop_ungrounded:
                    continue
                raise ProfileReadError(
                    "UngroundedProfileDetails",
                    "Some extracted details could not be matched to the document. Please clarify them in chat.",
                )
            grounded.append(excerpt)
        sections[field] = (grounded[0] if grounded else None) if field == "name" else grounded
    if sum(len(str(value)) for value in sections.values()) > 6000:
        raise ProfileReadError(
            "ProfileOutputLimit",
            "Too many profile details were returned. Please use a shorter document.",
        )
    return sections


def merge_profile(sources, chat_updates=()):
    def comparison_key(field, value):
        key = text_key(value)
        return key.rstrip(".") if field == "name" else key

    evidence = []
    conflicts = []
    references = []
    for source in sources:
        if source.get("status") == "reference":
            references.append({"source_id": source["source_id"], "url": source["source_url"]})
        for field in FIELDS:
            value = (source.get("facts") or {}).get(field)
            excerpts = [value] if field == "name" and value else value or []
            for excerpt in excerpts:
                evidence.append(
                    {
                        "fact_id": content_hash(
                            f"{source['source_id']}:{field}:{text_key(excerpt)}"
                        ),
                        "field": field,
                        "quote": excerpt,
                        "source_id": source["source_id"],
                        "origin": "source",
                        "status": "active",
                        "verification": "self_reported",
                    }
                )
    for update in chat_updates:
        for item in evidence:
            if item["fact_id"] in update.get("superseded_fact_ids", []):
                item.update(status="superseded", superseded_by_message=update["message_id"])
            if item["fact_id"] in update.get("confirmed_fact_ids", []):
                item.update(
                    verification="confirmed_by_candidate", confirmed_by_message=update["message_id"]
                )
        for field in FIELDS:
            value = update["facts"].get(field)
            excerpts = [value] if field == "name" and value else value or []
            for excerpt in excerpts:
                evidence.append(
                    {
                        "fact_id": content_hash(
                            f"{update['message_id']}:{field}:{text_key(excerpt)}"
                        ),
                        "field": field,
                        "quote": excerpt,
                        "source_id": update["message_id"],
                        "message_id": update["message_id"],
                        "request_id": update["request_id"],
                        "origin": "chat",
                        "status": "active",
                        "verification": "self_reported",
                        "context_subject": update.get("context_subject"),
                    }
                )
    result = {field: [] for field in FIELDS}
    active = [item for item in evidence if item["status"] == "active"]
    for item in active:
        values = result[item["field"]]
        if comparison_key(item["field"], item["quote"]) not in {
            comparison_key(item["field"], value) for value in values
        }:
            values.append(item["quote"])
    if len(result["name"]) > 1:
        conflicts.append({"field": "identity", "values": result["name"]})
    years = {}
    for item in active:
        matches = re.findall(
            r"(\d+(?:\.\d+)?)\s*\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?(?:Python|experience\s+(?:with|in)\s+Python)",
            item["quote"],
            flags=re.I,
        )
        matches += re.findall(
            r"Python\s+experience\s*[:=]?\s*(\d+(?:\.\d+)?)\s*\+?\s*years?",
            item["quote"],
            flags=re.I,
        )
        if (
            not matches
            and item.get("context_subject") == "python_experience"
            and item["field"] == "experience"
        ):
            matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", item["quote"], flags=re.I)
        for match in matches:
            years.setdefault(str(float(match)), []).append(item)
    if len(years) > 1:
        conflicts.append(
            {
                "field": "python_experience",
                "claims": [item for group in years.values() for item in group],
            }
        )
    missing = []
    if not years:
        missing.append("Explicit years of Python experience")
    if not result["skills"]:
        missing.append("Relevant skills")
    return {
        "sections": result,
        "evidence": evidence,
        "conflicts": conflicts,
        "missing_information": missing,
        "references": references,
        "chat_updates": list(chat_updates),
        "verification": {
            "unresolved_fields": [item["field"] for item in conflicts],
            "confirmed_fact_ids": [
                item["fact_id"]
                for item in active
                if item["verification"] == "confirmed_by_candidate"
            ],
            "scope": "candidate statements; not external credential verification",
        },
    }


def snapshot_fingerprint(source_ids):
    return content_hash(json.dumps(sorted(source_ids)))
