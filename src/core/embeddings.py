"""Single shared embedding model for the whole app.

Previously the agent, the ingestion script, and the semantic cache each built
their own ``HuggingFaceEmbeddings`` instance. This module builds it once
(prefers the fine-tuned domain model when present) and caches it so retriever
and cache share identical vectors. Heavy imports stay lazy so lightweight
consumers (and tests) can import this module without the ML stack installed.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Protocol, runtime_checkable

from src.core.config import get_settings


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface shared by LangChain embeddings and test fakes."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


@lru_cache(maxsize=1)
def get_embeddings() -> Embedder:
    """Return the shared embedder, preferring the fine-tuned model if available."""
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = get_settings().embeddings
    finetuned = get_settings().resolve(cfg.finetuned_path)
    if os.path.exists(finetuned):
        print("🧠 Loading fine-tuned domain embeddings...")
        return HuggingFaceEmbeddings(model_name=finetuned)
    print(f"ℹ️  Using base embedding model: {cfg.base_model}")
    return HuggingFaceEmbeddings(model_name=cfg.base_model)
