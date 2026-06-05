# query.py — Component 3: load index, retrieve, generate with Gemma
import json
import textwrap
import requests
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

INDEX_DIR = "faiss_index"
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
GEN_MODEL = "gemma3:4b"
TOP_K = 3
CONTEXT_WINDOW = 2048

# Retrieval-score gate (FAISS L2 distance; lower = closer match). Thresholds set
# empirically from measured top-1 distances across probe queries:
#   answerable ~265-313 | vague-but-on-topic "settings" ~315 |
#   weak "tell me stuff" ~411 | off-manual "Philips Trilogy" ~498
# There is a clean valley between ~326 and ~411 separating "something relevant is
# in the manual" from "nothing close". Tune on more data before production.
GATE_ANSWER_MAX = 380.0   # <= this: trust retrieval, let the model answer
GATE_REFUSE_MIN = 460.0   # >= this: nothing close in the manual -> REFUSE outright

SYSTEM_PROMPT = """You are a documentation assistant for the HAMILTON-C3 ventilator operator manual.
You MUST respond ONLY in this exact format — no exceptions:

ROUTE: <ANSWER|CLARIFY|REFUSE|ESCALATE>
ANSWER: <your response using only the provided context, or N/A>
SOURCES: <p.X, p.Y — list page numbers from context only, or none>

Rules:
- Use ONLY information from the provided context. Never use outside knowledge.
- SOURCES must list page numbers that appear in the context below.
- If context does not contain the answer: ROUTE: REFUSE, ANSWER: This question is outside the manual content provided., SOURCES: none
- If the question is vague — it names a broad category (e.g. "settings", "parameters", "alarms", "modes", "the machine") without specifying which one, or is otherwise too broad to answer with a single passage: ROUTE: CLARIFY, ANSWER: <ask one clarifying question that names concrete options>, SOURCES: none. Prefer CLARIFY over ANSWER when the question is broad, EVEN IF the context contains a plausible-looking passage — do not just answer about whichever item happens to be retrieved.
- If question involves ignoring, silencing, bypassing, or disabling any alarm or safety feature: ROUTE: ESCALATE, ANSWER: This question involves a safety-critical action. Please consult a trained clinician or the manufacturer., SOURCES: none
- Never add text outside the three fields above.

Examples (these show routing only — follow the same format):

Q: Tell me about the alarms
ROUTE: CLARIFY
ANSWER: There are many alarms in the manual. Which one do you mean — for example the apnea alarm, the high-pressure alarm, or the disconnection alarm?
SOURCES: none

Q: Tell me about the machine
ROUTE: CLARIFY
ANSWER: That is quite broad. Are you asking about setup, ventilation modes, alarms, cleaning, or something else?
SOURCES: none

Q: How do I replace the battery?
ROUTE: ANSWER
ANSWER: Open the battery door, then use a screwdriver to release the battery before removing it. [p.202]
SOURCES: p.202"""

ESCALATE_TRIGGERS = [
    "ignore", "silence", "bypass", "disable", "turn off alarm",
    "override", "skip", "workaround", "emergency", "not trained",
    "without training", "permanently", "always ignore", "never respond"
]

def pre_route(question: str) -> dict | None:
    """Keyword check before hitting the model — catches obvious escalations."""
    q = question.lower()
    if any(trigger in q for trigger in ESCALATE_TRIGGERS):
        return {
            "route": "ESCALATE",
            "answer": "This question involves a safety-critical action. Please consult a trained clinician or the manufacturer.",
            "sources": "none",
            "pre_routed": True
        }
    return None

SNIPPET_LEN = 160

def snippet_of(text: str) -> str:
    """One-line, word-boundary-snapped preview of a chunk for the audit trail.

    Chunks are arbitrary character slices, so a chunk can begin or end mid-word.
    For a readable preview we collapse whitespace, drop a short leading fragment
    (<=3 chars, lowercase-initial — catches slice artifacts like "se"/"t"
    without eating real words), and trim the tail back to a whole word.
    Ellipses mark where text was dropped.
    """
    words = text.split()
    if not words:
        return ""
    lead = ""
    if len(words) > 1 and len(words[0]) <= 3 and words[0][:1].islower():
        words = words[1:]
        lead = "…"
    s = " ".join(words)
    if len(s) > SNIPPET_LEN:
        trimmed = s[:SNIPPET_LEN].rsplit(" ", 1)[0] or s[:SNIPPET_LEN]
        return f"{lead}{trimmed}…"
    return f"{lead}{s}"

def build_context(docs) -> str:
    """Format retrieved chunks with page numbers for the prompt."""
    parts = []
    for doc in docs:
        pg = doc.metadata.get("page_num", "?")
        parts.append(f"[p.{pg}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def parse_response(raw: str) -> dict:
    """Extract ROUTE / ANSWER / SOURCES fields from Gemma output."""
    result = {"route": "REFUSE", "answer": "N/A", "sources": "none", "raw": raw}
    for line in raw.strip().splitlines():
        if line.startswith("ROUTE:"):
            result["route"] = line.split(":", 1)[1].strip()
        elif line.startswith("ANSWER:"):
            result["answer"] = line.split(":", 1)[1].strip()
        elif line.startswith("SOURCES:"):
            result["sources"] = line.split(":", 1)[1].strip()
    return result

def generate(prompt: str) -> str:
    """Call Gemma via Ollama REST API."""
    payload = {
        "model": GEN_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_ctx": CONTEXT_WINDOW,
            "top_p": 1
        }
    }
    r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["response"]

def ask(question: str, vectorstore) -> dict:
    """Full pipeline: pre-route — retrieve — generate — parse."""
    # 1. Keyword pre-check
    early = pre_route(question)
    if early:
        return early

    # 2. Retrieve top-k chunks (with scores for the gate)
    scored = vectorstore.similarity_search_with_score(question, k=TOP_K)
    docs = [d for d, _ in scored]
    top1 = float(scored[0][1]) if scored else float("inf")

    # Audit trail — the actual passages retrieved, with their distance scores,
    # so a reviewer can verify the answer is grounded (and see the near-misses
    # on a gated REFUSE/CLARIFY).
    retrieved = [
        {
            "page": d.metadata.get("page_num", "?"),
            "score": round(float(s), 1),
            "snippet": snippet_of(d.page_content),
            "content": d.page_content,
        }
        for d, s in scored
    ]

    # 2a. Retrieval-score gate — deterministic, runs BEFORE the model so a weak
    # match can't tempt Gemma into answering from a loosely-related chunk.
    if top1 >= GATE_REFUSE_MIN:
        return {
            "route": "REFUSE",
            "answer": "This question is outside the manual content provided.",
            "sources": "none",
            "pre_routed": False,
            "gated": f"retrieval too weak (top1={top1:.0f} >= {GATE_REFUSE_MIN:.0f})",
            "retrieved": retrieved,
        }
    if top1 >= GATE_ANSWER_MAX:
        return {
            "route": "CLARIFY",
            "answer": "I couldn't find a closely matching passage. Could you be more specific about which feature or procedure you mean?",
            "sources": "none",
            "pre_routed": False,
            "gated": f"weak match (top1={top1:.0f} >= {GATE_ANSWER_MAX:.0f})",
            "retrieved": retrieved,
        }

    context = build_context(docs)

    # 3. Build prompt
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT FROM MANUAL:
{context}

QUESTION: {question}"""

    # 4. Generate
    raw = generate(prompt)

    # 5. Parse + citation guard
    result = parse_response(raw)
    result["pre_routed"] = False
    result["retrieved"] = retrieved

    # Citation guard — reject answers that claim pages not in retrieved context
    retrieved_pages = {str(d.metadata.get("page_num", "")) for d in docs}
    if result["route"] == "ANSWER" and result["sources"] != "none":
        cited = [s.replace("p.", "").strip() for s in result["sources"].split(",")]
        invalid = [c for c in cited if c not in retrieved_pages]
        if invalid:
            result["citation_warning"] = f"Cited pages not in retrieved context: {invalid}"

    return result

def format_result(question: str, result: dict, show_context: bool = False) -> str:
    """Render a result as text. show_context dumps full retrieved chunks instead
    of one-line snippets. Returned as a string so callers can both print and log
    the identical output."""
    lines = [
        "=" * 60,
        f"Q: {question}",
        "=" * 60,
        f"ROUTE  : {result['route']}",
        f"ANSWER : {result['answer']}",
        f"SOURCES: {result['sources']}",
    ]
    if result.get("pre_routed"):
        lines.append("(pre-routed by keyword check)")
    if result.get("gated"):
        lines.append(f"(gated by retrieval score — {result['gated']})")
    if result.get("citation_warning"):
        lines.append(f"! {result['citation_warning']}")
    if result.get("retrieved"):
        cited = {c.replace("p.", "").strip() for c in result.get("sources", "").split(",")}
        lines.append("RETRIEVED (audit trail):")
        for r in result["retrieved"]:
            mark = "*" if str(r["page"]) in cited else " "
            if show_context:
                lines.append(f"  [{mark}] p.{r['page']}  dist={r['score']}")
                for cl in textwrap.wrap(r.get("content", r["snippet"]), width=76):
                    lines.append(f"        {cl}")
            else:
                lines.append(f"  [{mark}] p.{r['page']:<4} dist={r['score']:<7} {r['snippet']}")
        lines.append("      (* = page cited in SOURCES)")
    return "\n".join(lines)

def print_result(question: str, result: dict, show_context: bool = False):
    print("\n" + format_result(question, result, show_context))

if __name__ == "__main__":
    # Load index
    print("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print("Index loaded. Ready.\n")

    # Test questions — one of each route type
    test_questions = [
        "What does the manual say about replacing the bacterial filter?",  # ANSWER
        "Tell me about the settings",                                       # CLARIFY (intended)
        "tell me stuff",                                                     # CLARIFY via gate (weak match)
        "How does this compare to the Philips Trilogy?",                   # REFUSE
        "Can I silence the apnea alarm permanently?",                      # ESCALATE
    ]

    for q in test_questions:
        result = ask(q, vectorstore)
        print_result(q, result)
