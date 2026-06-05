# Ask the Manual — Day 1 Report

**Date:** 2026-06-04
**Project:** `ask-the-manual` (RAG over a medical-device operator manual)
**Status:** ✅ Environment ready, PDF source verified, extraction confirmed clean

---

## Summary

Day 1 setup is complete. The Python environment is built, all RAG dependencies
are installed, and the source PDF downloads and extracts cleanly. One blocker was
hit and resolved (the original PDF URL was dead) — details below.

---

## Step 0 — Repo name

Chosen: **`ask-the-manual`** (per recommendation). Folder created at
`/home/hari/ask-the-manual`. Easy to rename later if needed.

---

## Step 1 — Environment

> Note: the kickoff script used Windows activation (`venv\Scripts\activate`).
> This machine is **Linux**, so commands were adapted (`venv/bin/...`). Result is
> identical.

- **Python:** 3.12.3 (`/usr/bin/python3`)
- **Virtualenv:** `venv/` created
- **Dependencies installed** (key versions confirmed):

  | Package | Version |
  |---|---|
  | langchain | 1.3.4 |
  | langchain-community | 0.4.2 |
  | langchain-anthropic | 1.4.4 |
  | faiss-cpu | 1.14.2 |
  | pypdf | 6.12.2 |
  | anthropic | 0.105.2 |
  | python-dotenv | 1.2.2 |
  | requests | 2.34.2 |

- **`.env`** created with placeholder `ANTHROPIC_API_KEY=your_key_here`
  — ⚠️ **ACTION NEEDED: paste your real key here.**
- **`.gitignore`** created (`.env`, `venv/`, `__pycache__/`, `*.faiss`, `*.pkl`)

---

## Step 2 — PDF download + extraction smoke test

### Blocker found & fixed
The PDF URL in the kickoff (`jcr:2955b97c-...Hamilton-C3-Operator-Manual-EN.pdf`)
is a **soft-404**: it returned HTTP 200 but the body was Hamilton's
"404 page not found" HTML page (98 KB, `content-type: text/html`). The DAM asset
ID was stale. `pypdf` correctly rejected it (`invalid pdf header: b'<!doc'`).

This was **not** a parsing problem, so swapping to `pdfplumber` would not have
helped — the fix was finding a live source.

### Working source (verified 2026-06-04)
```
https://www.hamilton-medical.com/dam/jcr:5687919f-6926-4268-aa7c-f935b513fc5b/HAMILTON-C3-ops-manual-SW2.0.x-en-624446.03.pdf
```
- `content-type: application/pdf`, 3,516,960 bytes, magic bytes `%PDF-`

### Smoke test result ✅
```
Status: 200 | Size: 3,516,960 bytes
Pages: 372

--- Page 1 ---
Intelligent Ventilation since 1983
HAMILTON-C3
160005 REF
Operator's Manual  624446/04 | 2021-01-12
Software version 2.0.x

--- Page 10 ---
10 English | 624446/04

--- Page 50 ---
2 Preparing for ventilation ... Position airway adapters with windows in a
vertical, not a horizontal, position...
```

**Verdict:** Text extraction is **clean** (real selectable text, not an image
scan). `pypdf` is the right tool — no `pdfplumber` needed.

---

## What this tells us for the build

- **372 pages** of clean text. At a typical ~1,000-char chunk with ~150 overlap,
  expect very roughly **1,500–2,500 chunks** (will confirm once we measure actual
  character count per page).
- Page headers/footers like `10 English | 624446/04` are boilerplate noise —
  worth stripping during chunking so they don't pollute retrieval.
- No OCR pipeline required; we go straight to load → split → embed → FAISS.

---

## Files created

```
ask-the-manual/
├── .env            # placeholder key — REPLACE before next step
├── .gitignore
├── test_pdf.py     # smoke test (now points at the working URL)
├── venv/
└── DAY1_REPORT.md  # this file
```

---

## Next steps (Day 2)

1. **Add your real API key** to `.env`.
2. Load the full PDF → split into chunks (measure real chunk count).
3. Embed chunks → build the FAISS index, persist to disk.
4. Wire retrieval + `langchain-anthropic` (Claude) → first end-to-end Q&A.
5. Add a simple query loop / CLI for the demo.
