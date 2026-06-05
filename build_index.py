# build_index.py — Component 2: chunk, embed, build FAISS index
import json
import time
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document  # langchain 1.x: was langchain.schema

PAGES_FILE = "pages.json"
INDEX_DIR = "faiss_index"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"

def chunk_pages(pages: list[dict]) -> list[Document]:
    """Slice pages into overlapping chunks, preserving page_num metadata."""
    docs = []
    for page in pages:
        text = page["text"]
        page_num = page["page_num"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end].strip()
            if chunk:
                docs.append(Document(
                    page_content=chunk,
                    metadata={"page_num": page_num}
                ))
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return docs

def verify_ollama():
    """Confirm Ollama is up before we start."""
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        assert any(EMBED_MODEL in m for m in models), \
            f"{EMBED_MODEL} not found. Run: ollama pull {EMBED_MODEL}"
        print(f"Ollama OK — {EMBED_MODEL} confirmed")
    except Exception as e:
        raise RuntimeError(f"Ollama not reachable: {e}")

if __name__ == "__main__":
    verify_ollama()

    # Load pages
    with open(PAGES_FILE) as f:
        pages = json.load(f)
    print(f"Loaded {len(pages)} pages from {PAGES_FILE}")

    # Chunk
    docs = chunk_pages(pages)
    print(f"Chunks created: {len(docs)}")
    print(f"  Settings: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    # Spot-check
    print(f"\n--- Chunk 0 ---")
    print(f"  Page: {docs[0].metadata['page_num']}")
    print(f"  Text: {docs[0].page_content[:120]}...")
    print(f"\n--- Chunk 100 ---")
    print(f"  Page: {docs[100].metadata['page_num']}")
    print(f"  Text: {docs[100].page_content[:120]}...")

    # Embed + build FAISS index
    print(f"\nEmbedding {len(docs)} chunks via {EMBED_MODEL}...")
    print("(Watch your GPU — should be near 100% utilization)")
    embeddings = OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE
    )

    t0 = time.time()
    vectorstore = FAISS.from_documents(docs, embeddings)
    elapsed = time.time() - t0

    print(f"Done in {elapsed:.1f}s")
    print(f"Throughput: {len(docs)/elapsed:.1f} chunks/sec")

    # Save to disk
    vectorstore.save_local(INDEX_DIR)
    print(f"\nIndex saved to {INDEX_DIR}/")
    print("This file is permanent — never rebuild unless the manual changes.")
