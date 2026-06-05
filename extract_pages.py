# extract_pages.py — Component 1.5: PDF -> pages.json
# Day 1 only smoke-tested 3 pages in memory. This extracts ALL pages to disk
# so the index build (Component 2) reads from a stable, versioned artifact
# instead of re-downloading the PDF every run.
import json
import os
import re
import io
import requests
from pypdf import PdfReader

PDF_URL = "https://www.hamilton-medical.com/dam/jcr:5687919f-6926-4268-aa7c-f935b513fc5b/HAMILTON-C3-ops-manual-SW2.0.x-en-624446.03.pdf"
PDF_CACHE = "manual.pdf"
PAGES_FILE = "pages.json"

# Boilerplate footer/header lines flagged as retrieval noise in DAY1_REPORT.md,
# e.g. "10 English | 624446/04". Strip only full-line matches so we never touch
# real body text. Conservative on purpose.
FOOTER_RE = re.compile(r"^\s*\d*\s*English\s*\|\s*\d+/\d+\s*$", re.IGNORECASE)


def clean(text: str) -> str:
    lines = [ln for ln in text.splitlines() if not FOOTER_RE.match(ln)]
    return "\n".join(lines).strip()


def get_pdf_bytes() -> bytes:
    if os.path.exists(PDF_CACHE):
        print(f"Using cached PDF: {PDF_CACHE}")
        with open(PDF_CACHE, "rb") as f:
            return f.read()
    print("Downloading PDF...")
    r = requests.get(PDF_URL, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    assert r.content[:5] == b"%PDF-", "Not a PDF — soft-404 again?"
    with open(PDF_CACHE, "wb") as f:
        f.write(r.content)
    print(f"Saved {len(r.content):,} bytes to {PDF_CACHE}")
    return r.content


if __name__ == "__main__":
    reader = PdfReader(io.BytesIO(get_pdf_bytes()))
    pages = []
    empty = 0
    for i, page in enumerate(reader.pages):
        text = clean(page.extract_text() or "")
        if not text:
            empty += 1
            continue
        pages.append({"page_num": i + 1, "text": text})

    with open(PAGES_FILE, "w") as f:
        json.dump(pages, f)

    total_chars = sum(len(p["text"]) for p in pages)
    print(f"Extracted {len(pages)} non-empty pages ({empty} blank skipped)")
    print(f"Total characters: {total_chars:,}")
    print(f"Avg chars/page: {total_chars // max(len(pages), 1):,}")
    print(f"Saved -> {PAGES_FILE}")
