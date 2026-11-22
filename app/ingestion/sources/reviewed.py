"""Fetch only individually reviewed employer URLs; never discover new postings."""

from io import BytesIO
from urllib.parse import urlsplit

from pypdf import PdfReader

from app.ingestion.description import html_markdown, markdown_text, pdf_markdown, text_key
from app.ingestion.logos import logo_candidates
from app.ingestion.normalize import digest, html_text, normalized_text
from app.ingestion.sources.boards import BoardResult


class SourceReviewRequired(ValueError):
    """The pinned employer wording changed and needs a deliberate review."""


def extract_description(payload, posting):
    if posting.get("format") == "pdf":
        text = normalized_text(
            "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(payload)).pages)
        )
    else:
        text = html_text(payload.decode("utf-8"))
    start, end = posting.get("start_marker"), posting.get("end_marker")
    if start:
        index = text.find(start)
        if index < 0:
            # A valid career page with no selected role is a successful absent-role scan.
            return None
        text = text[index:]
    if end:
        index = text.find(end, len(start or ""))
        if index < 0:
            raise ValueError("Reviewed job section boundary changed; manual review is required")
        text = text[:index]
    text = normalized_text(text)
    if len(text) < 200:
        raise ValueError("Selected job description is empty or incomplete")
    if posting.get("format") == "pdf":
        return pdf_markdown(BytesIO(payload), posting["title"])
    return html_markdown(
        payload,
        title=posting["title"],
        start_marker=posting.get("start_marker"),
        end_marker=posting.get("end_marker"),
        base_url=posting.get("url", ""),
    )


def fetch_reviewed(source, client, *, retries=2):
    # Each request downloads the full content of one explicitly selected posting.
    postings = source.get("postings", [])
    if len(postings) != 1:
        raise ValueError("Reviewed sources must contain exactly one pinned posting")
    posting = postings[0]
    url = posting["url"]
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Reviewed sources require a public HTTPS URL without credentials")
    import time

    import httpx

    for attempt in range(retries + 1):
        try:
            response = client.get(url, follow_redirects=True)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()
            break
        except httpx.TransportError:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    if len(response.content) > 25_000_000:
        raise ValueError("Source exceeds the 25 MB document limit")
    if urlsplit(str(response.url)).netloc != parsed.netloc:
        raise ValueError("Employer URL redirected to another host; manual review is required")
    description = extract_description(response.content, posting)
    if description is None:
        return BoardResult([], set())
    expected = posting.get("reviewed_text_sha256")
    if expected and digest(text_key(markdown_text(description))) != expected:
        raise SourceReviewRequired(
            "Reviewed job wording changed; retain the published capture until the source is reviewed"
        )
    source_id = f"web:{source['board']}"
    external_id = posting["external_job_id"]
    title = posting["title"]
    if not all(word.lower() in description.lower() for word in ("python",)):
        raise ValueError("Selected section is no longer a Python job description")
    job = {
        "job_id": "job-" + digest(f"{source_id}:{external_id}")[:24],
        "source_id": source_id,
        "external_job_id": external_id,
        "canonical_url": url,
        "company_name": source["company"],
        "title": title,
        "description_text": description,
        "description_format": "markdown",
        "locations": [{"location_text": value} for value in posting.get("locations", [])],
        "employment_type": posting.get("employment_type"),
        "workplace_type": posting.get("workplace_type"),
        "seniority": posting.get("seniority"),
        "compensation": posting.get("compensation"),
        "source_published_at": posting.get("source_published_at"),
        "source_updated_at": None,
        "description_notices": posting.get("source_notes", []),
    }
    if posting.get("format") == "pdf":
        job["original_pdf"] = response.content
    logos = []
    if posting.get("format") != "pdf":
        try:
            logos = logo_candidates(response.text, str(response.url), source["company"])
        except (ValueError, TypeError, RecursionError):
            pass
    return BoardResult([job], {external_id}, logos)
