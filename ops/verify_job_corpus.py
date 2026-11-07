"""Validate all PDF content/layout and render pages for visual inspection (dev dependencies)."""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ingestion.description import validate_pdf_description  # noqa: E402


def canonical(text):
    return re.sub(r"[\W_]", "", unicodedata.normalize("NFKC", text)).lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "config/job_corpus_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/ingestion/pdf-qa")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    policy = json.loads((ROOT / "config/job_sources.json").read_text())
    native_pdf_urls = {
        posting["url"]
        for source in policy["sources"]
        for posting in source.get("postings", [])
        if posting.get("format") == "pdf"
    }
    args.output.mkdir(parents=True, exist_ok=True)
    pages_dir = args.output / "pages"
    pages_dir.mkdir(exist_ok=True)
    results, thumbs = [], []
    for record in manifest["jobs"]:
        path = ROOT / record["pdf_path"]
        errors, body, extracted_pages, page_count = [], [], [], 0
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["pdf_sha256"]:
            errors.append("PDF changed since ingestion")
        with pymupdf.open(path) as document:
            for index, page in enumerate(document):
                page_count += 1
                extracted_pages.append((index + 1, page.get_text()))
                # Rendering every page catches unreadable/corrupt font resources as well.
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(0.75, 0.75), alpha=False)
                image_path = pages_dir / f"{record['job_id']}-{index + 1}.png"
                pixmap.save(image_path)
                if index == 0:
                    thumbs.append((record["job_id"], image_path))
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        for span in line["spans"]:
                            x0, y0, x1, y1 = span["bbox"]
                            if (
                                x0 < -1
                                or y0 < -1
                                or x1 > page.rect.width + 1
                                or y1 > page.rect.height + 1
                            ):
                                errors.append(f"page {index + 1}: text outside page")
                            if y0 < page.rect.height - 45:
                                body.append(span["text"])
                            if "\ufffd" in span["text"] or "\x00" in span["text"]:
                                errors.append(f"page {index + 1}: replacement glyph")
        if record.get("description_format") == "markdown":
            try:
                validate_pdf_description(record["description_text"], extracted_pages)
            except ValueError:
                errors.append("catalog source description differs from extracted PDF body")
        elif record["document_id"].startswith("posting-"):
            expected = canonical(record["description_text"])
            actual = canonical(" ".join(body))
            if expected not in actual:
                errors.append("catalog source description differs from extracted PDF body")
        if record["document_id"].startswith("posting-"):
            actual = canonical(" ".join(body))
            if (
                record["canonical_url"] not in native_pdf_urls
                and canonical(record["canonical_url"]) not in actual
            ):
                errors.append("missing source attribution")
        results.append(
            {
                "job_id": record["job_id"],
                "pdf_path": record["pdf_path"],
                "pages": page_count,
                "errors": sorted(set(errors)),
            }
        )
    # A contact sheet for each group of 20 first pages supports whole-corpus visual QA.
    for offset in range(0, len(thumbs), 20):
        sheet = Image.new("RGB", (1000, 1500), "#dbe3eb")
        draw = ImageDraw.Draw(sheet)
        for index, (job_id, path) in enumerate(thumbs[offset : offset + 20]):
            col, row = index % 4, index // 4
            thumb = Image.open(path)
            thumb.thumbnail((236, 270))
            x, y = col * 250 + 7, row * 300 + 20
            sheet.paste(thumb, (x, y))
            draw.text((x, row * 300 + 5), job_id[:25], fill="black")
        sheet.save(args.output / f"contact-{offset // 20 + 1}.png")
    report = {
        "documents": len(results),
        "pages": sum(item["pages"] for item in results),
        "failures": sum(bool(item["errors"]) for item in results),
        "companies": dict(Counter(record["company_name"] for record in manifest["jobs"])),
        "checks": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in ("documents", "pages", "failures", "companies")}, indent=2
        )
    )
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
