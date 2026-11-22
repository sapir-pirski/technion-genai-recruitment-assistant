"""Source-aware descriptions shared by collection, PDF generation and the UI."""

import re
import unicodedata
from urllib.parse import urljoin, urlsplit

import pymupdf
from bs4 import BeautifulSoup, Comment
from markdown_it import MarkdownIt
from markdownify import markdownify

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


def _anchor(soup, marker, *, after=-1, title=""):
    strings = list(soup.find_all(string=True))
    positions = {id(node): index for index, node in enumerate(strings)}
    first = marker.strip().splitlines()[0]
    if first.startswith("Home / Careers /"):
        first = title
    matches = [
        node
        for node in strings
        if positions[id(node)] > after and text_key(str(node)) == text_key(first)
    ]
    if not matches:
        raise ValueError("Reviewed job section boundary changed; manual review is required")
    if after < 0:
        headings = [node for node in matches if node.find_parent(re.compile(r"^h[1-6]$"))]
        if headings:
            matches = headings
    return positions[id(matches[0])]


def _semantic_cards(soup):
    """Preserve explicitly numbered cards and check-mark lists used by employers."""
    for parent in list(soup.find_all("div")):
        if parent.parent is None:
            continue
        cards = parent.find_all("article", recursive=False)
        if len(cards) >= 2 and all(card.find(re.compile(r"^h[1-6]$")) for card in cards):
            numbers = [
                card.find(string=lambda text: text and re.fullmatch(r"\s*\d{2}\s*", text))
                for card in cards
            ]
            if all(numbers):
                ordered = soup.new_tag("ol")
                for card, number in zip(cards, numbers, strict=True):
                    number.extract()
                    item = soup.new_tag("li")
                    for heading in card.find_all(re.compile(r"^h[1-6]$")):
                        heading.name = "span"
                    item.string = _text(card.get_text(" ", strip=True))
                    ordered.append(item)
                parent.replace_with(ordered)
                continue
        cards = parent.find_all("div", recursive=False)
        if len(cards) >= 2 and all(
            len(card.select("svg.lucide-check")) == 1 and len(card.find_all("p")) == 1
            for card in cards
        ):
            unordered = soup.new_tag("ul")
            for card in cards:
                item = soup.new_tag("li")
                for child in list(card.find("p").contents):
                    item.append(child.extract())
                unordered.append(item)
            parent.replace_with(unordered)


def html_markdown(payload, *, title, start_marker=None, end_marker=None, base_url=""):
    soup = BeautifulSoup(payload, "html.parser")
    for tag in list(soup.find_all(["script", "style", "noscript", "iframe"])):
        tag.decompose()
    for node in soup.find_all(string=lambda text: isinstance(text, Comment)):
        node.extract()
    if start_marker:
        start = _anchor(soup, start_marker, title=title)
        end = (
            _anchor(soup, end_marker, after=start)
            if end_marker
            else len(list(soup.find_all(string=True)))
        )
        for index, node in enumerate(list(soup.find_all(string=True))):
            if index < start or index >= end:
                node.extract()
    _semantic_cards(soup)
    for tag in list(soup.find_all(["svg", "img", "input", "button", "nav", "footer", "aside"])):
        tag.decompose()
    for tag in reversed(list(soup.find_all(True))):
        if tag.parent is not None and tag.name != "br" and not tag.get_text(strip=True):
            tag.decompose()
    for heading in list(soup.find_all(re.compile(r"^h[1-6]$"))):
        if not _title_line(heading.get_text(" ", strip=True), title):
            continue
        for parent in list(heading.parents):
            if parent.name == "li" and parent.parent and parent.parent.name in {"ul", "ol"}:
                listing = parent.parent
                if len(listing.find_all("li", recursive=False)) == 1:
                    parent.unwrap()
                    listing.unwrap()
    for tag in soup.find_all("div"):
        classes = " ".join(tag.get("class", []))
        spans = tag.find_all("span", recursive=False)
        if (
            re.search(r"meta|badge|chip", classes, re.IGNORECASE)
            and len(spans) >= 2
            and len(tag.get_text()) <= 160
        ):
            value = " · ".join(_text(span.get_text(" ", strip=True)) for span in spans)
            if text_key(tag.get_text(" ", strip=True)) == text_key(
                " ".join(span.get_text(" ", strip=True) for span in spans)
            ):
                tag.clear()
                tag.string = value
    for anchor in list(soup.find_all("a")):
        if CTA.fullmatch(_text(anchor.get_text(" ", strip=True))):
            anchor.decompose()
            continue
        if anchor.get("href", "").startswith("#"):
            anchor.unwrap()
            continue
        href = urljoin(base_url, anchor.get("href", ""))
        parsed = urlsplit(href)
        if (
            parsed.fragment
            and parsed._replace(fragment="").geturl()
            == urlsplit(base_url)._replace(fragment="").geturl()
        ):
            anchor.unwrap()
            continue
        if urlsplit(href).scheme not in {"https", "http", "mailto"}:
            anchor.unwrap()
        else:
            anchor["href"] = href
    for term in list(soup.find_all("dt")):
        definition = term.find_next_sibling("dd")
        if definition:
            paragraph = soup.new_tag("p")
            paragraph.string = (
                _text(term.get_text(" ", strip=True))
                + ": "
                + _text(definition.get_text(" ", strip=True))
            )
            term.replace_with(paragraph)
            definition.decompose()
    for paragraph in list(soup.find_all("p")):
        leading = paragraph.find(["strong", "b"], recursive=False)
        if leading and HEADING.match(_text(leading.get_text())) and len(leading.get_text()) < 100:
            before = "".join(str(node) for node in leading.previous_siblings).strip()
            rest = (
                paragraph.get_text(" ", strip=True)
                .removeprefix(leading.get_text(" ", strip=True))
                .strip()
            )
            if not before and rest:
                heading = soup.new_tag("h2")
                heading.string = leading.get_text(" ", strip=True)
                heading["data-description-heading"] = "promoted"
                paragraph.insert_before(heading)
                leading.decompose()
                for child in list(paragraph.children):
                    if getattr(child, "name", None) == "br":
                        child.decompose()
                    elif str(child).strip():
                        break
        text = _text(paragraph.get_text(" ", strip=True))
        if not text or len(text) > 100 or paragraph.find_parent(["li", "td", "th"]):
            continue
        if "eyebrow" in paragraph.get("class", []):
            strong = soup.new_tag("strong")
            for child in list(paragraph.contents):
                strong.append(child.extract())
            paragraph.append(strong)
            continue
        if (HEADING.match(text) and not re.search(r"[.!?]", text.rstrip(".?"))) or (
            paragraph.find(["strong", "b"], recursive=False)
            and _text(paragraph.get_text()) == _text(paragraph.find(["strong", "b"]).get_text())
            and not _title_line(text, title)
        ):
            paragraph.name = "h2"
            paragraph["data-description-heading"] = "promoted"
    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    for tag in headings:
        if re.match(
            r"^(?:Location|Worksite|Salary|Posted|Experience|Employment|Workplace)(?:\s*:|$)",
            tag.get_text(" ", strip=True),
            re.IGNORECASE,
        ):
            tag.name = "p"
    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    levels = [
        int(tag.name[1])
        for tag in headings
        if not _title_line(tag.get_text(), title) and not tag.get("data-description-heading")
    ]
    minimum = min(levels, default=2)
    for tag in headings:
        if tag.get("data-description-heading"):
            level = (
                3
                if tag.get_text()
                .strip()
                .casefold()
                .startswith(("core technical", "additional tools"))
                else 2
            )
        else:
            level = max(2, min(4, int(tag.name[1]) - minimum + 2))
        tag.name = "h" + str(level)
    value = markdownify(
        str(soup), heading_style="ATX", bullets="-", wrap=False, newline_style="SPACES"
    )
    return clean_markdown(value, title)


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
    return BeautifulSoup(MARKDOWN.render(value), "html.parser").get_text(" ", strip=True)


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
