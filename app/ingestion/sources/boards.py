import json
import time
from dataclasses import dataclass, field

import httpx

from app.ingestion.description import clean_markdown, html_markdown
from app.ingestion.logos import LogoCandidate
from app.ingestion.normalize import digest


@dataclass
class BoardResult:
    jobs: list[dict]
    all_ids: set[str]
    logo_candidates: list[LogoCandidate] = field(default_factory=list)


def board_url(source: dict) -> str:
    provider, board = source["provider"], source["board"]
    if not board.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Invalid board identifier")
    if provider == "greenhouse":
        return f"https://boards-api.greenhouse.io/v1/boards/{board}/jobs?content=true"
    if provider == "lever":
        return f"https://api.lever.co/v0/postings/{board}?mode=json"
    if provider == "ashby":
        return f"https://api.ashbyhq.com/posting-api/job-board/{board}?includeCompensation=true"
    raise ValueError("Unsupported job provider")


def normalize_job(source: dict, item: dict) -> dict:
    provider = source["provider"]
    if provider == "greenhouse":
        external_id = str(item["id"])
        url, title = item["absolute_url"], item["title"]
        text = html_markdown(item.get("content", ""), title=title, base_url=url)
        locations = [{"location_text": item.get("location", {}).get("name", "Unknown")}]
        workplace, employment, published = None, None, None
        updated, compensation = item.get("updated_at"), None
    elif provider == "lever":
        external_id, url, title = str(item["id"]), item["hostedUrl"], item["text"]
        parts = [
            html_markdown(item.get("description", ""), title=title, base_url=url)
            or item.get("descriptionPlain", "")
        ]
        for section in item.get("lists", []):
            parts.extend(
                [
                    "## " + section.get("text", ""),
                    html_markdown(section.get("content", ""), title=title, base_url=url),
                ]
            )
        parts.append(
            html_markdown(item.get("additional", ""), title=title, base_url=url)
            or item.get("additionalPlain", "")
        )
        text = clean_markdown("\n\n".join(parts), title)
        categories = item.get("categories", {})
        locations = [{"location_text": name} for name in categories.get("allLocations", [])]
        if not locations:
            locations = [{"location_text": categories.get("location", "Unknown")}]
        workplace, employment = item.get("workplaceType"), categories.get("commitment")
        published, updated, compensation = None, None, item.get("salaryRange")
    else:
        url, title = item["jobUrl"], item["title"]
        external_id = str(item.get("id") or url.rstrip("/").split("/")[-1])
        text = html_markdown(
            item.get("descriptionHtml", ""), title=title, base_url=url
        ) or clean_markdown(item.get("descriptionPlain", ""), title)
        locations = [{"location_text": item.get("location", "Unknown")}]
        locations.extend(
            {"location_text": loc["location"]} for loc in item.get("secondaryLocations", [])
        )
        workplace, employment = item.get("workplaceType"), item.get("employmentType")
        published, updated, compensation = item.get("publishedAt"), None, item.get("compensation")
    if not url.startswith("https://") or len(text) < 200:
        raise ValueError("Posting has no usable URL or description")
    source_id = f"{provider}:{source['board']}"
    return {
        "job_id": "job-" + digest(f"{source_id}:{external_id}")[:24],
        "source_id": source_id,
        "external_job_id": external_id,
        "canonical_url": url.split("?")[0],
        "company_name": source["company"],
        "title": title,
        "description_text": text,
        "description_format": "markdown",
        "locations": locations,
        "workplace_type": workplace,
        "employment_type": employment,
        "seniority": None,
        "compensation": compensation,
        "source_published_at": published,
        "source_updated_at": updated,
        "raw": item,
    }


def fetch_board(source: dict, client: httpx.Client, *, retries=2) -> BoardResult:
    url = board_url(source)
    for attempt in range(retries + 1):
        try:
            response = client.get(url)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                delay = response.headers.get("Retry-After", "")
                time.sleep(min(float(delay), 15) if delay.isdigit() else 2**attempt)
                continue
            response.raise_for_status()
            break
        except httpx.TransportError:
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    payload = response.json()
    provider = source["provider"]
    rows = payload if provider == "lever" else payload["jobs"]
    if not isinstance(rows, list):
        raise ValueError("Invalid board response")
    jobs, all_ids = [], set()
    for row in rows:
        # Ashby unlisted postings are deliberately excluded.
        if provider == "ashby" and not row.get("isListed", True):
            continue
        key = str(row.get("id") or row.get("jobUrl", "").rstrip("/").split("/")[-1])
        if not key or key == "None":
            raise ValueError("Incomplete board identity; refusing closure reconciliation")
        all_ids.add(key)
        try:
            jobs.append(normalize_job(source, row))
        except (ValueError, KeyError, TypeError):
            # Still present in all_ids, so malformed descriptions cannot close known jobs.
            continue
    if provider == "greenhouse" and payload.get("meta", {}).get("total", len(rows)) != len(rows):
        raise ValueError("Incomplete board response")
    if provider == "ashby" and payload.get("apiVersion") not in (None, "1"):
        raise ValueError("Unknown Ashby response version")
    return BoardResult(jobs, all_ids)


def load_sources(path, *, include_disabled=False):
    values = json.loads(path.read_text())
    if values.get("version") != 1:
        raise ValueError("Unsupported source configuration version")
    return [s for s in values["sources"] if include_disabled or s.get("enabled", True)]


def selected_job_ids(settings):
    """Return explicitly selected source identities, including the assignment role."""
    from app.ingestion.models import DEFAULT_JOB_ID

    if not settings.job_sources_path.exists():
        return None
    policy = json.loads(settings.job_sources_path.read_text())
    if policy.get("selection_mode") != "reviewed_only":
        return None
    return {
        DEFAULT_JOB_ID,
        *(job_id for source in policy["sources"] for job_id in source.get("selected_job_ids", [])),
    }
