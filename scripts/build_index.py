"""Build the knowledge index from one or more PDFs.

Usage:
    python scripts/build_index.py path/to/doc.pdf [more.pdf ...]

Replaces the old hardcoded-``./new.pdf`` ``src/utils/ingest.py``. Extracts
(LlamaParse → PyPDF fallback), cleans/splits, builds a RAPTOR tree, and writes
every node into LanceDB.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from src.agent.nodes import get_store
from src.core.config import get_settings
from src.data.ingestion import extract_documents_from_pdf
from src.index.builder import index_documents

load_dotenv()


def main(paths: list[str]) -> None:
    if not paths:
        print("Usage: python scripts/build_index.py <file.pdf> [more.pdf ...]")
        raise SystemExit(1)

    store = get_store()
    cfg = get_settings().retriever
    total = 0
    for path in paths:
        print(f"📄 Ingesting {path} ...")
        docs = extract_documents_from_pdf(path)
        total += index_documents(docs, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap, store=store)
    print(f"✅ Indexed {total} nodes. Store now holds {store.count()} nodes.")


if __name__ == "__main__":
    main(sys.argv[1:])
