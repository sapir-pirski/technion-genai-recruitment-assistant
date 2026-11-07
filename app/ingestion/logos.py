"""Employer identity and bounded, cached collection of public company logos."""

import ipaddress
import json
import logging
import re
import socket
import time
import warnings
from base64 import b64decode
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import unquote_to_bytes, urljoin, urlsplit
from xml.etree import ElementTree

import httpx
from PIL import Image, ImageOps, ImageStat
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingestion.models import Company, Job, JobSource
from app.ingestion.normalize import digest
from app.observability.logging import log_event

MAX_DOWNLOAD = 2_000_000
MAX_PIXELS = 4_000_000
LOGO_SIDE = 192
LOGO_PIPELINE_VERSION = "logos-v1"


@dataclass(frozen=True)
class LogoCandidate:
    url: str
    data: bytes | None = None
    kind: str = "logo"


def _words(value):
    return set(re.findall(r"[^\W_]+", value.casefold())) if isinstance(value, str) else set()


class LogoParser(HTMLParser):
    """Prefer employer organization metadata and branding, not arbitrary page images."""

    def __init__(self, base_url, company_name):
        super().__init__(convert_charrefs=True)
        self.base_url, self.company_words = base_url, _words(company_name)
        self.stack, self.ranked, self.structured = [], [], []
        self.json_parts = None
        self.svg_parts = None
        self.svg_depth = 0

    def candidate(self, url, score, data=None, kind="logo"):
        if not isinstance(url, str) or not url:
            return
        if url.startswith("data:image/") and len(url) <= MAX_DOWNLOAD:
            try:
                kind, encoded = url.split(",", 1)
                data = (
                    b64decode(encoded, validate=True)
                    if ";base64" in kind
                    else unquote_to_bytes(encoded)
                )
                url = self.base_url
            except ValueError:
                return
        if len(url) <= 2048:
            self.ranked.append((score, LogoCandidate(urljoin(self.base_url, url), data, kind)))

    def handle_starttag(self, tag, attrs):
        attrs = {key: value or "" for key, value in attrs}
        context = " ".join(str(value) for _, value in self.stack[-3:]).lower()
        label = " ".join(attrs.get(key, "") for key in ("alt", "aria-label", "title"))
        brand = bool(self.company_words and self.company_words <= _words(label))
        own = " ".join(
            attrs.get(key, "") for key in ("id", "class", "src", "data-framer-name")
        ).lower()
        branding = "logo" in own and any(
            word in context for word in ("header", "nav", "logo", "brand")
        )
        if self.svg_parts is not None:
            self.svg_parts.append(self.get_starttag_text())
            self.svg_depth += 1
        elif tag == "svg" and (brand or branding or "logo" in context):
            self.svg_parts, self.svg_depth = [self.get_starttag_text()], 1
        excluded = {
            "partner",
            "client-logo",
            "ccpa",
            "certif",
            "badge",
            "sponsor",
            "twitter",
            "facebook",
            "linkedin",
            "instagram",
            "youtube",
            "social",
        } - self.company_words
        badge = any(word in (context + " " + label + " " + own).lower() for word in excluded)
        if tag == "img" and not badge and (brand or branding or "logo" in context):
            url = attrs.get("data-src") or attrs.get("src")
            score = 100 if brand else 80
            if any(word in (url or "").lower() for word in ("white", "blanc", "light")):
                score -= 35
            self.candidate(url, score)
        if not badge and (branding or "logo" in context):
            for url in re.findall(
                r"background-image\s*:\s*url\(['\"]?(.*?)['\"]?\)", attrs.get("style", ""), re.I
            ):
                self.candidate(url, 80)
        if tag == "link" and "icon" in attrs.get("rel", "").lower():
            self.candidate(
                attrs.get("href"), 30 if "apple" in attrs.get("rel", "") else 20, kind="icon"
            )
        if tag == "script" and attrs.get("type", "").lower() == "application/ld+json":
            self.json_parts = []
        if tag not in {
            "img",
            "link",
            "meta",
            "br",
            "input",
            "hr",
            "source",
            "path",
            "use",
            "circle",
            "rect",
            "polygon",
            "line",
            "ellipse",
            "polyline",
            "stop",
        }:
            self.stack.append(
                (
                    tag,
                    " ".join(
                        [
                            tag,
                            attrs.get("class", ""),
                            attrs.get("id", ""),
                            attrs.get("data-framer-name", ""),
                        ]
                    ),
                )
            )

    def handle_startendtag(self, tag, attrs):
        if self.svg_parts is not None:
            self.svg_parts.append(self.get_starttag_text())
        else:
            self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag):
        if self.svg_parts is not None:
            self.svg_parts.append(f"</{tag}>")
            self.svg_depth -= 1
            if self.svg_depth == 0:
                self.candidate(self.base_url, 90, "".join(self.svg_parts).encode())
                self.svg_parts = None
        if tag == "script" and self.json_parts is not None:
            try:
                self.structured.append(json.loads("".join(self.json_parts)))
            except (ValueError, TypeError, RecursionError):
                pass
            self.json_parts = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, value):
        if self.json_parts is not None:
            self.json_parts.append(value)
        if self.svg_parts is not None:
            self.svg_parts.append(value)

    def results(self):
        def walk(value, depth=0):
            if depth > 20:
                return
            if isinstance(value, list):
                for item in value:
                    walk(item, depth + 1)
            elif isinstance(value, dict):
                kinds = value.get("@type", [])
                kinds = kinds if isinstance(kinds, list) else [kinds]
                if any(kind in {"Organization", "Corporation"} for kind in kinds):
                    if self.company_words and self.company_words <= _words(value.get("name", "")):
                        logo = value.get("logo")
                        self.candidate(
                            logo.get("url") or logo.get("contentUrl")
                            if isinstance(logo, dict)
                            else logo,
                            120,
                        )
                for item in value.values():
                    if isinstance(item, (list, dict)):
                        walk(item, depth + 1)

        for value in self.structured:
            walk(value)
        output, seen = [], set()
        for _, candidate in sorted(self.ranked, key=lambda item: -item[0]):
            key = (candidate.url, digest(candidate.data) if candidate.data else None)
            if key not in seen and (candidate.data or candidate.url.startswith("https://")):
                output.append(candidate)
                seen.add(key)
        return output[:8]


def logo_candidates(html, base_url, company_name):
    parser = LogoParser(base_url, company_name)
    parser.feed(html)
    return parser.results()


def public_addresses(host):
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        addresses = {
            ipaddress.ip_address(item[4][0])
            for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        }
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Logo URLs must resolve to public addresses")
    return addresses


def fetch_public(client, url, settings, *, headers=None):
    """Bound every response and validate each HTTPS redirect before following it."""
    for _ in range(4):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Logo URLs require public HTTPS without credentials")
        public_addresses(parsed.hostname)
        request_headers = {"Accept-Encoding": "identity", **(headers or {})}
        for attempt in range(settings.max_retries + 1):
            try:
                with client.stream(
                    "GET",
                    url,
                    headers=request_headers,
                    follow_redirects=False,
                    timeout=min(settings.request_timeout, 10),
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        url = urljoin(url, response.headers["location"])
                        break
                    if (
                        response.status_code in {429, 500, 502, 503, 504}
                        and attempt < settings.max_retries
                    ):
                        time.sleep(min(2**attempt, 3))
                        continue
                    if response.status_code == 304:
                        return response.status_code, b"", dict(response.headers), url
                    response.raise_for_status()
                    if int(response.headers.get("content-length", 0)) > MAX_DOWNLOAD:
                        raise ValueError("Logo response is too large")
                    parts, size = [], 0
                    for part in response.iter_bytes(chunk_size=65536):
                        size += len(part)
                        if size > MAX_DOWNLOAD:
                            raise ValueError("Logo response is too large")
                        parts.append(part)
                    return response.status_code, b"".join(parts), dict(response.headers), url
            except httpx.TransportError:
                if attempt >= settings.max_retries:
                    raise
                time.sleep(min(2**attempt, 3))
        else:
            raise ValueError("Logo request did not complete")
    raise ValueError("Too many logo redirects")


def normalize_logo(payload):
    if len(payload) > MAX_DOWNLOAD:
        raise ValueError("Logo image is too large")
    if payload.lstrip().startswith(b"<") and re.search(rb"<svg\b", payload[:4096], re.I):
        if re.search(rb"<!DOCTYPE|<!ENTITY|<\?xml-stylesheet", payload, re.I):
            raise ValueError("SVG entities are not allowed")
        root = ElementTree.fromstring(payload)
        if root.tag.rsplit("}", 1)[-1] != "svg":
            raise ValueError("Invalid SVG")
        identifiers = {node.attrib["id"] for node in root.iter() if "id" in node.attrib}
        for index, node in enumerate(root.iter()):
            if index > 10000 or node.tag.rsplit("}", 1)[-1].lower() in {
                "script",
                "foreignobject",
                "image",
                "feimage",
                "style",
            }:
                raise ValueError("Unsupported SVG content")
            for key, value in node.attrib.items():
                if (
                    key.lower().startswith("on")
                    or (key.rsplit("}", 1)[-1] == "href" and not value.startswith("#"))
                    or any(
                        not target.strip().strip("'\"").startswith("#")
                        for target in re.findall(r"url\((.*?)\)", value, re.I)
                    )
                ):
                    raise ValueError("External SVG references are not allowed")
                if key.rsplit("}", 1)[-1] == "href" and value[1:] not in identifiers:
                    raise ValueError("SVG references an unavailable symbol")
        # MuPDF does not reproduce class-based SVG styles. Use another candidate
        # instead of silently changing an employer's artwork.
        import pymupdf

        with pymupdf.open(stream=payload, filetype="svg") as document:
            page = document[0]
            ratio = LOGO_SIDE / max(page.rect.width, page.rect.height)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(ratio, ratio), alpha=True)
            image = Image.frombytes("RGBA", (pixmap.width, pixmap.height), pixmap.samples)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as source:
                if source.width * source.height > MAX_PIXELS or min(source.size) < 8:
                    raise ValueError("Invalid logo dimensions")
                image = ImageOps.exif_transpose(source).convert("RGBA")
    if image.getbbox() is None:
        raise ValueError("Logo image is empty")
    if all(
        low > 245
        for low, _ in ImageStat.Stat(image.convert("RGB"), mask=image.getchannel("A")).extrema
    ):
        raise ValueError("White-only logo is not visible on the light background")
    image = ImageOps.contain(image, (LOGO_SIDE, LOGO_SIDE), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (LOGO_SIDE, LOGO_SIDE))
    canvas.alpha_composite(image, ((LOGO_SIDE - image.width) // 2, (LOGO_SIDE - image.height) // 2))
    output = BytesIO()
    canvas.save(output, format="PNG", optimize=True)
    if len(output.getvalue()) > 256_000:
        raise ValueError("Normalized logo is too large")
    return output.getvalue()


def _refresh(company, sources, hints, client, settings):
    allow_icons = all(source.get("logo_favicon_fallback", True) for source in sources)
    candidates = [item for item in hints if allow_icons or item.kind != "icon"]
    for source in sources:
        if source.get("logo_url"):
            candidates.insert(0, LogoCandidate(source["logo_url"]))
    errors, pages = [], []
    if not candidates:
        if company.website_url:
            pages.append(company.website_url)
        pages.extend(
            posting["url"]
            for source in sources
            for posting in source.get("postings", [])
            if posting.get("format") != "pdf"
        )
        for page in dict.fromkeys(pages):
            try:
                _, data, _, final = fetch_public(client, page, settings)
                candidates = logo_candidates(
                    data.decode("utf-8", errors="replace"), final, company.name
                )
                candidates = [item for item in candidates if allow_icons or item.kind != "icon"]
                if candidates:
                    break
            except Exception as error:
                errors.append(error)
    if (
        allow_icons
        and company.logo_source_url
        and not any(item.url == company.logo_source_url for item in candidates)
    ):
        candidates.append(LogoCandidate(company.logo_source_url))
    for candidate in list(dict.fromkeys(candidates))[:8]:
        try:
            headers = {}
            if (
                company.logo_data
                and candidate.url == company.logo_source_url
                and candidate.data is None
                and company.logo_pipeline_version == LOGO_PIPELINE_VERSION
            ):
                if company.logo_etag:
                    headers["If-None-Match"] = company.logo_etag
                if company.logo_last_modified:
                    headers["If-Modified-Since"] = company.logo_last_modified
            if candidate.data is not None:
                status, raw, response_headers = 200, candidate.data, {}
            else:
                status, raw, response_headers, _ = fetch_public(
                    client, candidate.url, settings, headers=headers
                )
            if status == 304:
                if not company.logo_data:
                    raise ValueError("Uncached logo returned not modified")
                return "unchanged", {"logo_status": "ready", "logo_error_code": None}
            image = normalize_logo(raw)
            sha = digest(image)
            fields = {
                "logo_data": image,
                "logo_mime_type": "image/png",
                "logo_sha256": sha,
                "logo_pipeline_version": LOGO_PIPELINE_VERSION,
                "logo_source_url": candidate.url,
                "logo_etag": response_headers.get("etag"),
                "logo_last_modified": response_headers.get("last-modified"),
                "logo_status": "ready",
                "logo_error_code": None,
            }
            if sha != company.logo_sha256:
                fields["logo_updated_at"] = datetime.now(UTC).replace(tzinfo=None)
            return "unchanged" if sha == company.logo_sha256 else "updated", fields
        except Exception as error:
            errors.append(error)
    if errors:
        return "failed", {
            "logo_status": "stale" if company.logo_data else "failed",
            "logo_error_code": type(errors[-1]).__name__,
        }
    return "missing", {"logo_status": "missing", "logo_error_code": None}


def refresh_company_logos(settings, registry, client, *, sources, hints=None, force=False):
    """Run under the ingestion lease. Refresh once per UTC day, separately from RAG."""
    counts, grouped = Counter(), {}
    with Session(registry.engine) as session:
        for source in sources:
            if not source.get("enabled", True):
                continue
            record = session.get(JobSource, f"{source['provider']}:{source['board']}")
            if (
                record
                and record.company_id
                and session.scalar(
                    select(Job.job_id)
                    .where(Job.company_id == record.company_id, Job.status == "active")
                    .limit(1)
                )
            ):
                grouped.setdefault(record.company_id, []).append(source)
        companies = list(session.scalars(select(Company).where(Company.company_id.in_(grouped))))
        session.expunge_all()
    checked = datetime.now(UTC).replace(tzinfo=None)
    for company in companies:
        candidates = [
            item
            for source in grouped[company.company_id]
            for item in (hints or {}).get(f"{source['provider']}:{source['board']}", [])
        ]
        if (
            not force
            and company.logo_checked_at
            and company.logo_checked_at.date() >= checked.date()
            and (not company.logo_data or company.logo_pipeline_version == LOGO_PIPELINE_VERSION)
        ):
            counts["logos_skipped"] += 1
            continue
        action, fields = _refresh(
            company, grouped[company.company_id], candidates, client, settings
        )
        with Session(registry.engine) as session, session.begin():
            record = session.get(Company, company.company_id)
            for key, value in {**fields, "logo_checked_at": checked}.items():
                setattr(record, key, value)
        counts["logos_checked"] += 1
        counts[f"logos_{action}"] += 1
        if action == "failed":
            log_event(
                "company_logo_refresh_failed",
                component="app.ingestion",
                level=logging.WARNING,
                company_id=company.company_id,
                error_code=fields["logo_error_code"],
            )
    return dict(counts)


def company_identity(source):
    if source["provider"] == "local":
        return None, None
    website = source.get("website_url")
    if not website and source["provider"] == "web":
        postings = source.get("postings", [])
        hosts = {urlsplit(posting["url"]).netloc for posting in postings}
        if len(hosts) == 1:
            website = f"https://{next(iter(hosts))}/"
    if website:
        parsed = urlsplit(website)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Company websites must use HTTPS without credentials")
        host = parsed.hostname.lower().rstrip(".")
        identity = "domain:" + host.removeprefix("www.")
        return identity, f"https://{host}/"
    # A shared ATS domain must never merge unrelated employers.
    return f"provider:{source['provider']}:{source['board']}", None
