"""Extract and display the assignment PDF's description without rewriting it."""

import re
import unicodedata

import pymupdf
from markdown_it import MarkdownIt

from app.ingestion.normalize import html_text

FORMAT_VERSION = "descriptions-v2"
MARKDOWN = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
CTA = re.compile(
    r"^(?:apply (?:now|here|for this (?:position|role|job))|see all roles|submit)\s*[»→]*$",
    re.IGNORECASE,
)
HEADING = re.compile(
    r"^(?:the opportunity|about\b|objectives\b|your (?:tasks|impact|key responsibilities)\b|"
    r"(?:key )?responsibilities\b|(?:required|preferred|core technical) (?:skills|qualifications)\b|"
    r"requirements\b|qualifications\b|technical stack\b|technology stack\b|"
    r"(?:good|nice)[- ]to[- ]have\b|must have\b|benefits\b|what (?:you|we)\b|"
    r"application process\b|role overview\b|job summary\b|skills (?:required|and attributes)\b|"
    r"additional tools\b|why join us\b|our perfect candidate\b|inclusiveness at\b)",
    re.IGNORECASE,
)


def text_key(value):
    """Compare wording across whitespace, punctuation and typographic variants."""
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold()))


def _text(value):
    return " ".join(value.replace("\u2060", "").replace("\u200b", "").split())


def _title_line(value, title):
    value = re.sub(r"^\s*(?:#{1,6}\s*|\d+[.)]\s*)", "", value)
    value = re.sub(r"^job title\s*:\s*", "", value, flags=re.IGNORECASE).strip(" *_")
    key, name = text_key(value), text_key(title)
    return key in {name, name + " job description", "job description"}


def clean_markdown(value, title):
    """Remove page furniture and exact repeated introductions, never rewrite prose."""
    lines = value.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    kept = []
    at_start = True
    opening = True
    pending_label = None
    labels = {"location", "worksite", "salary", "posted", "experience", "employment", "workplace"}
    for line in lines:
        line = line.rstrip().replace("\u2060", "").replace("\u200b", "")
        plain = re.sub(r"[*_#]", "", line).strip()
        if CTA.fullmatch(plain):
            continue
        if opening and _title_line(plain, title):
            continue
        if at_start and (
            not plain or _title_line(plain, title) or plain.startswith("Home / Careers /")
        ):
            continue
        if at_start and re.fullmatch(
            r"(?:JD\s+NO\.?\s*\d+|[A-Z]+-\d+(?:-\d+)+)", plain, re.IGNORECASE
        ):
            kept.extend(["**Posting ID:** " + plain, ""])
            continue
        if plain:
            at_start = False
            if len(plain) > 160 or re.match(r"^[-*+]\s", line) or HEADING.match(plain):
                opening = False
        if text_key(plain) in labels:
            pending_label = plain.rstrip(":")
            continue
        if pending_label and plain:
            kept.extend([f"**{pending_label}:** {plain}", ""])
            pending_label = None
            continue
        metadata = re.fullmatch(
            r"(Location|Worksite|Salary|Posted|Experience|Employment|Workplace):\s*(.+)",
            plain,
            re.IGNORECASE,
        )
        if metadata:
            kept.append(f"**{metadata[1]}:** {metadata[2]}")
            continue
        badges = re.fullmatch(
            r"(.+?)\s+(Full[- ]time|Part[- ]time|Contract)\s+(Senior|Junior|Mid[- ]level)",
            plain,
            re.IGNORECASE,
        )
        if badges:
            kept.append(" · ".join(badges.groups()))
            continue
        kept.append(line)
    if pending_label:
        kept.append(pending_label)
    value = "\n".join(kept).strip()
    blocks = re.split(r"\n\s*\n", value)
    narrative_counts = {}
    for block in blocks:
        key = text_key(block)
        if len(key) > 300 and not re.match(r"^(?:#|[-*+] |\d+\.)", block):
            narrative_counts[key] = narrative_counts.get(key, 0) + 1
    result = []
    for block in blocks:
        key = text_key(block)
        # Only long, identical narrative paragraphs are deduplicated. Lists and
        # headings may legitimately repeat under different source sections.
        narrative = len(key) > 300 and not re.match(r"^(?:#|[-*+] |\d+\.)", block)
        if narrative:
            narrative_counts[key] -= 1
            if narrative_counts[key]:
                continue
        result.append(block)
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(result)).strip()


def plain_markdown(value, title):
    """Conservative fallback for genuine plain text; explicit bullet characters survive."""
    lines = []
    for raw in value.splitlines():
        line = _text(raw)
        if not line:
            continue
        bullet = re.match(r"^[•◦▪‣●]\s*(.*)", line)
        if bullet:
            lines.append("- " + bullet[1])
        elif (
            len(line) <= 100 and HEADING.match(line) and not re.search(r"[.!?]", line.rstrip(".?"))
        ):
            lines.extend(["", "## " + line, ""])
        else:
            lines.extend([line, ""])
    return clean_markdown("\n".join(lines), title)


def pdf_markdown(path, title):
    """Recover source headings, paragraph wraps and visible list indentation."""
    value = []
    document = (
        pymupdf.open(stream=path.getvalue(), filetype="pdf")
        if hasattr(path, "getvalue")
        else pymupdf.open(path)
    )
    try:
        first_page = document[0].get_text()
        snapshot_header = (
            "PUBLIC JOB POSTING SNAPSHOT |" in first_page
            and "Generated snapshot of a real public posting." in first_page
        )
        for page_number, page in enumerate(document):
            for block in page.get_text("dict", sort=True)["blocks"]:
                if "lines" not in block:
                    continue
                lines = [
                    line
                    for line in block["lines"]
                    if any(span["text"].strip() for span in line["spans"])
                ]
                if not lines:
                    continue
                text = _text(" ".join(span["text"] for line in lines for span in line["spans"]))
                if text.startswith("PUBLIC JOB POSTING SNAPSHOT |") or (
                    text.isdigit() and block["bbox"][1] > page.rect.height - 45
                ):
                    continue
                if snapshot_header and page_number == 0:
                    if text.startswith("Generated snapshot of a real public posting."):
                        snapshot_header = False
                    continue
                pending = ""
                for line in lines:
                    spans = [span for span in line["spans"] if span["text"].strip()]
                    text = _text(" ".join(span["text"] for span in spans))
                    if not text:
                        continue
                    bullet = re.match(r"^(?:[•◦▪‣●]|[-*](?=\s))\s*(.*)", text)
                    if bullet:
                        if pending:
                            value.append(pending)
                        pending = "- " + bullet[1]
                    elif len(text) <= 160 and all(span["flags"] & 16 for span in spans):
                        if pending:
                            value.extend([pending, ""])
                            pending = ""
                        value.extend(["## " + text, ""])
                    else:
                        pending = (pending + " " + text).strip()
                if pending:
                    value.extend([pending, ""])
    finally:
        document.close()
    return clean_markdown("\n".join(value), title)


def description_html(value, *, title="", format="markdown"):
    markdown = value if format == "markdown" else plain_markdown(value, title)
    return (
        '<article class="job-description" aria-label="Full job description">'
        + MARKDOWN.render(clean_markdown(markdown, title))
        + "</article>"
    )


def markdown_text(value):
    """Semantic comparison and unsupported-character validation, without markup."""
    return html_text(MARKDOWN.render(value)).replace("\n", " ")


def presentation_for(job, markdown, *, notices=None):
    fields = (
        "title",
        "company_name",
        "canonical_url",
        "employment_type",
        "seniority",
        "workplace_type",
        "compensation",
        "source_published_at",
    )
    metadata = {key: job.get(key) for key in fields}
    metadata["locations"] = [
        location if isinstance(location, str) else location["location_text"]
        for location in job.get("locations", [])
    ]
    return {
        "format_version": FORMAT_VERSION,
        "description_text": clean_markdown(markdown, job["title"]),
        "description_format": "markdown",
        "job": metadata,
        "notices": notices if notices is not None else job.get("description_notices", []),
    }


def validate_pdf_description(markdown, pages):
    """Every displayed block must be present in the PDF used for retrieval."""

    def compact(text):
        return "".join(
            char for char in unicodedata.normalize("NFKC", text).casefold() if char.isalnum()
        )

    content = []
    for _, text in pages:
        lines = text.splitlines()
        content.extend(
            line
            for line in lines
            if not line.startswith("PUBLIC JOB POSTING SNAPSHOT |") and not line.isdigit()
        )
    document = compact(" ".join(content))
    for token in MARKDOWN.parse(markdown):
        if token.type == "inline":
            visible = markdown_text(token.content)
            if visible.startswith("Posting ID:"):
                visible = visible.removeprefix("Posting ID:").strip()
            block = compact(visible)
            if block and block not in document:
                raise ValueError("Displayed description differs from the PDF; refusing publication")
