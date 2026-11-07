"""Attributed PDFs generated from selected public job descriptions."""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.ingestion.description import MARKDOWN, clean_markdown, markdown_text, plain_markdown


def display(text):
    for char in ("\u200b", "\ufeff"):
        text = text.replace(char, "")
    for char in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        text = text.replace(char, "-")
    return text


def inline_markup(token):
    parts = []
    for child in token.children or []:
        if child.type in {"text", "code_inline", "html_inline"}:
            parts.append(escape(display(child.content)))
        elif child.type in {"strong_open", "strong_close"}:
            parts.append("<b>" if child.nesting == 1 else "</b>")
        elif child.type in {"em_open", "em_close"}:
            parts.append("<i>" if child.nesting == 1 else "</i>")
        elif child.type == "link_open":
            href = escape(child.attrGet("href") or "", {'"': "&quot;"})
            parts.append(f'<link href="{href}" color="#1d4ed8">')
        elif child.type == "link_close":
            parts.append("</link>")
        elif child.type in {"softbreak", "hardbreak"}:
            parts.append(" " if child.type == "softbreak" else "<br/>")
    return "".join(parts)


def markdown_flowables(value, body, *, width=499):
    """Render the same Markdown hierarchy as the UI, including nested lists."""
    tokens = MARKDOWN.parse(value)
    heading = {
        level: ParagraphStyle(
            f"Heading{level}",
            parent=body,
            fontName="JobBold",
            fontSize=12 if level == 2 else 10.5,
            leading=17 if level == 2 else 15,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )
        for level in (2, 3, 4)
    }

    def blocks(index=0, until=None):
        result = []
        while index < len(tokens):
            token = tokens[index]
            if token.type == until:
                return result, index + 1
            if token.type in {"bullet_list_open", "ordered_list_open"}:
                ordered = token.type == "ordered_list_open"
                close = "ordered_list_close" if ordered else "bullet_list_close"
                start = int(token.attrGet("start") or 1)
                items = []
                index += 1
                while index < len(tokens) and tokens[index].type != close:
                    if tokens[index].type == "list_item_open":
                        item, index = blocks(index + 1, "list_item_close")
                        items.append(ListItem(item, spaceAfter=3))
                    else:
                        index += 1
                result.append(
                    ListFlowable(
                        items,
                        bulletType="1" if ordered else "bullet",
                        start=start if ordered else "bulletchar",
                        leftIndent=18,
                        bulletFontName="JobText",
                        bulletFontSize=9,
                        spaceBefore=1,
                        spaceAfter=6,
                    )
                )
                index += 1
            elif token.type in {"paragraph_open", "heading_open"}:
                style = (
                    heading.get(max(2, min(4, int(token.tag[1:]))), body)
                    if token.type == "heading_open"
                    else body
                )
                if token.type == "heading_open":
                    result.append(CondPageBreak(80))
                result.append(Paragraph(inline_markup(tokens[index + 1]), style))
                index += 3
            elif token.type == "table_open":
                rows = []
                index += 1
                while index < len(tokens) and tokens[index].type != "table_close":
                    if tokens[index].type == "tr_open":
                        rows.append([])
                    elif tokens[index].type == "inline" and rows:
                        rows[-1].append(Paragraph(inline_markup(tokens[index]), body))
                    index += 1
                if rows:
                    table = Table(
                        rows, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1
                    )
                    table.setStyle(
                        TableStyle(
                            [
                                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eff6ff")),
                                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dbe3ee")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ]
                        )
                    )
                    result.extend([table, Spacer(1, 8)])
                index += 1
            elif token.type in {"fence", "code_block"}:
                result.append(
                    Paragraph(escape(display(token.content)).replace("\n", "<br/>"), body)
                )
                index += 1
            else:
                index += 1
        return result, index

    return blocks()[0]


def make_snapshot(job: dict, path: Path, captured_at: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fonts = Path(__file__).parent / "fonts"
    for name, filename in (
        ("JobText", "DejaVuSans.ttf"),
        ("JobBold", "DejaVuSans-Bold.ttf"),
        ("JobItalic", "DejaVuSans-Oblique.ttf"),
        ("JobBoldItalic", "DejaVuSans-BoldOblique.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
    pdfmetrics.registerFontFamily(
        "JobText", normal="JobText", bold="JobBold", italic="JobItalic", boldItalic="JobBoldItalic"
    )
    pdfmetrics.registerFontFamily(
        "JobBold",
        normal="JobBold",
        bold="JobBold",
        italic="JobBoldItalic",
        boldItalic="JobBoldItalic",
    )
    font = pdfmetrics.getFont("JobText")
    markdown = (
        clean_markdown(job["description_text"], job["title"])
        if job.get("description_format") == "markdown"
        else plain_markdown(job["description_text"], job["title"])
    )
    content = job["title"] + markdown_text(markdown) + job["company_name"]
    unsupported = {
        c for c in display(content) if ord(c) > 31 and ord(c) not in font.face.charToGlyph
    }
    if unsupported:
        raise ValueError(
            "Snapshot font cannot represent source characters; add a suitable licensed font"
        )
    body = ParagraphStyle(
        "JobBody",
        fontName="JobText",
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
        alignment=TA_LEFT,
        splitLongWords=True,
        allowWidows=0,
        allowOrphans=0,
    )
    title = ParagraphStyle(
        "JobTitle", parent=body, fontName="JobBold", fontSize=19, leading=25, spaceAfter=14
    )
    meta = ParagraphStyle(
        "JobMeta", parent=body, fontSize=8, leading=12, textColor=colors.HexColor("#42546a")
    )
    story = [Paragraph(escape(display(job["title"])), title)]
    locations = "; ".join(loc["location_text"] for loc in job.get("locations", []))
    for label, value in (
        ("Company", job["company_name"]),
        ("Location", locations or None),
        ("Workplace", job.get("workplace_type")),
        ("Employment", job.get("employment_type")),
        ("Seniority", job.get("seniority")),
        ("Salary", (job.get("compensation") or {}).get("display")),
        ("Posted", job.get("source_published_at")),
        ("Source", job["canonical_url"]),
        ("Posting ID", job["external_job_id"]),
        ("Captured (UTC)", captured_at),
    ):
        if value is not None:
            story.append(Paragraph(f"{label}: {escape(display(str(value)))}", meta))
    story.append(
        Paragraph(
            "Generated snapshot of a real public posting. This is not an employer-issued PDF. The description below preserves the captured source text; availability may change.",
            meta,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.extend(markdown_flowables(markdown, body))
    for notice in job.get("description_notices", []):
        story.append(Paragraph("Source note: " + escape(display(notice)), meta))
    document = SimpleDocTemplate(
        str(path),
        pagesize=(595.28, 841.89),
        rightMargin=48,
        leftMargin=48,
        topMargin=48,
        bottomMargin=48,
        title=job["title"],
        author=job["company_name"],
    )

    def footer(canvas, doc):
        canvas.setFont("JobText", 7)
        canvas.setFillColor(colors.HexColor("#42546a"))
        canvas.drawString(48, 27, "PUBLIC JOB POSTING SNAPSHOT | Source attribution on page 1")
        canvas.drawRightString(547, 27, str(doc.page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
