"""Indexing pipeline: documents → RAPTOR tree → LanceDB.

Single place that ties together preprocessing, hierarchical summarization, and
storage. Used by both ``scripts/build_index.py`` and the Streamlit uploader so
ingestion behaves identically everywhere.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.agent import prompts
from src.agent.nodes import get_llm, get_store
from src.data.preprocessing import prepare_documents
from src.index.raptor import build_raptor_nodes
from src.index.vectorstore import KnowledgeStore


def default_summarizer() -> Callable[[str], str]:
    """LLM summarizer used to build RAPTOR's higher levels."""
    chain = prompts.SUMMARIZE_CLUSTER | get_llm("grade") | StrOutputParser()
    return lambda passages: chain.invoke({"passages": passages})


def index_chunks(
    chunks: Sequence[Document],
    store: Optional[KnowledgeStore] = None,
    summarize: Optional[Callable[[str], str]] = None,
) -> int:
    """Build a RAPTOR tree over chunks and write every node to the store."""
    store = store or get_store()
    summarize = summarize or default_summarizer()

    leaves = [{"content": c.page_content, "metadata": dict(c.metadata or {})} for c in chunks]
    nodes = build_raptor_nodes(leaves, store.embedder, summarize)
    if not nodes:
        return 0
    store.add(
        texts=[n["content"] for n in nodes],
        metadatas=[n["metadata"] for n in nodes],
        vectors=[n["vector"] for n in nodes],
    )
    return len(nodes)


def index_documents(
    docs: Sequence[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    store: Optional[KnowledgeStore] = None,
) -> int:
    """Clean + split raw documents, then index them. Returns nodes written."""
    chunks = prepare_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return index_chunks(chunks, store=store)
