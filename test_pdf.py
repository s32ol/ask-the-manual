# test_pdf.py — PDF download + text extraction smoke test
import requests
from pypdf import PdfReader
import io

# Official Hamilton Medical DAM link (verified live 2026-06-04). The asset ID
# in the original kickoff URL was stale and soft-404'd, so it was replaced.
PDF_URL = "https://www.hamilton-medical.com/dam/jcr:5687919f-6926-4268-aa7c-f935b513fc5b/HAMILTON-C3-ops-manual-SW2.0.x-en-624446.03.pdf"

print("Downloading PDF...")
response = requests.get(PDF_URL, timeout=45, headers={"User-Agent": "Mozilla/5.0"})
print(f"Status: {response.status_code} | Size: {len(response.content):,} bytes")

reader = PdfReader(io.BytesIO(response.content))
print(f"Pages: {len(reader.pages)}")

# Spot-check pages 1, 10, 50
for pg_num in [0, 9, 49]:
    text = reader.pages[pg_num].extract_text() or ""
    print(f"\n--- Page {pg_num+1} (first 200 chars) ---")
    print(text[:200].strip())
