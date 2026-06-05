# chat.py — Component 4: interactive CLI loop for the demo.
# Reuses the exact pipeline from query.py (pre-route -> gate -> retrieve ->
# Gemma -> parse + citation guard) so the demo and the test harness can never
# drift apart. Loads the FAISS index once, then loops on stdin.
#
# Flags:
#   --show-context   dump full retrieved chunks (not just snippets)
#
# Every session is logged to sessions/session_<timestamp>.log for a reviewable
# record (the same text shown on screen, including the audit trail).
import os
import sys
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

from query import (
    INDEX_DIR, OLLAMA_BASE, EMBED_MODEL,
    ask, format_result,
)

EXIT_WORDS = {"exit", "quit", ":q", "q"}
SESSION_DIR = "sessions"

BANNER = """\
Ask the Manual — HAMILTON-C3 operator manual (local RAG)
Type a question and press Enter. Type 'exit' or 'quit' (or Ctrl-D) to leave.
"""


def main():
    show_context = "--show-context" in sys.argv[1:]

    print("Loading FAISS index...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE)
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )
    print("Index loaded.\n")

    # Open a timestamped session log.
    os.makedirs(SESSION_DIR, exist_ok=True)
    started = datetime.now()
    log_path = os.path.join(SESSION_DIR, f"session_{started:%Y%m%d_%H%M%S}.log")
    log = open(log_path, "a", encoding="utf-8")
    log.write(f"# Ask-the-Manual session — {started.isoformat(timespec='seconds')}\n")
    log.write(f"# show_context={show_context}\n\n")
    log.flush()

    print(f"Logging session to {log_path}")
    if show_context:
        print("(--show-context: full retrieved chunks will be shown)")
    print()
    print(BANNER)

    try:
        while True:
            try:
                question = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break

            if not question:
                continue
            if question.lower() in EXIT_WORDS:
                print("Bye.")
                break

            try:
                result = ask(question, vectorstore)
                text = format_result(question, result, show_context=show_context)
            except Exception as e:
                text = f"[error] {type(e).__name__}: {e}"

            print("\n" + text + "\n")
            log.write(f"[{datetime.now():%H:%M:%S}] {text}\n\n")
            log.flush()
    finally:
        log.write(f"# session ended — {datetime.now().isoformat(timespec='seconds')}\n")
        log.close()
        print(f"Session log saved to {log_path}")


if __name__ == "__main__":
    main()
