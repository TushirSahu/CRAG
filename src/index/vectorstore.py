"""LanceDB-backed knowledge store with native hybrid search.

Replaces the previous Chroma + hand-rolled ``CustomEnsembleRetriever`` (BM25 +
manual RRF). LanceDB gives us, in one dependency:

* native **hybrid** retrieval (vector + full-text) with a built-in RRF reranker,
* SQL-style ``where`` filters used here for **temporal** (`as_of`) queries and
  **source ACL**, and
* an embedded, on-disk store (no server) that is multimodal- and version-ready.

The store is intentionally decoupled from LangChain: it accepts any
:class:`~src.core.embeddings.Embedder` (real model or a test fake).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import lancedb

from src.core.config import Settings, get_settings
from src.core.embeddings import Embedder

# Columns promoted to first-class fields; everything else rides in ``meta_json``.
_RESERVED = {"source", "page", "level", "node_type", "valid_from", "ingested_at"}
_DAY_SECONDS = 86_400


def _to_where(as_of: Optional[float], sources: Optional[Sequence[str]]) -> Optional[str]:
    clauses: List[str] = []
    if as_of is not None:
        clauses.append(f"valid_from <= {float(as_of)}")
    if sources:
        joined = ", ".join("'" + s.replace("'", "''") + "'" for s in sources)
        clauses.append(f"source IN ({joined})")
    return " AND ".join(clauses) if clauses else None


class KnowledgeStore:
    """Thin wrapper over a single LanceDB table of knowledge nodes."""

    def __init__(self, embedder: Embedder, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.embedder = embedder
        cfg = self.settings.vector_store
        self._db = lancedb.connect(self.settings.resolve(cfg.path))
        self._table_name = cfg.table

    # -- writes -----------------------------------------------------------
    def add(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
        vectors: Optional[Sequence[Sequence[float]]] = None,
    ) -> int:
        """Embed (unless ``vectors`` supplied) and upsert nodes. Returns count added."""
        if not texts:
            return 0
        metadatas = list(metadatas or [{} for _ in texts])
        if vectors is None:
            vectors = self.embedder.embed_documents(list(texts))

        now = time.time()
        rows: List[Dict[str, Any]] = []
        for text, meta, vec in zip(texts, metadatas, vectors):
            extra = {k: v for k, v in meta.items() if k not in _RESERVED}
            rows.append(
                {
                    "id": str(meta.get("id", uuid.uuid4())),
                    "text": text,
                    "vector": list(vec),
                    "source": str(meta.get("source", "unknown")),
                    "page": int(meta.get("page", 0) or 0),
                    "level": int(meta.get("level", 0)),
                    "node_type": str(meta.get("node_type", "leaf")),
                    "valid_from": float(meta.get("valid_from", now)),
                    "ingested_at": float(meta.get("ingested_at", now)),
                    "meta_json": json.dumps(extra),
                }
            )

        if self._table_name in self._db.table_names():
            self._db.open_table(self._table_name).add(rows)
        else:
            self._db.create_table(self._table_name, data=rows)
        self._ensure_fts(force=True)
        return len(rows)

    def reset(self) -> None:
        if self._table_name in self._db.table_names():
            self._db.drop_table(self._table_name)

    def count(self) -> int:
        if self._table_name not in self._db.table_names():
            return 0
        return self._db.open_table(self._table_name).count_rows()

    def texts(self, limit: Optional[int] = None, leaves_only: bool = True) -> List[str]:
        """Return stored chunk texts (leaf nodes by default) — e.g. for eval/data gen."""
        if self._table_name not in self._db.table_names():
            return []
        rows = self._db.open_table(self._table_name).to_arrow().to_pylist()
        if leaves_only:
            rows = [r for r in rows if r.get("node_type") == "leaf"]
        out = [r["text"] for r in rows]
        return out[:limit] if limit else out

    # -- reads ------------------------------------------------------------
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        as_of: Optional[float] = None,
        sources: Optional[Sequence[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid (vector + full-text) search with optional temporal / ACL filters."""
        if self._table_name not in self._db.table_names() or not query.strip():
            return []

        rcfg = self.settings.retriever
        k = k or rcfg.k_retrieval
        table = self._db.open_table(self._table_name)
        where = _to_where(as_of, sources)
        qvec = self.embedder.embed_query(query)

        try:
            if rcfg.hybrid and self._ensure_fts():
                from lancedb.rerankers import RRFReranker

                builder = (
                    table.search(query_type="hybrid")
                    .vector(qvec)
                    .text(query)
                    .rerank(reranker=RRFReranker())
                )
            else:
                builder = table.search(qvec)
            if where:
                builder = builder.where(where, prefilter=True)
            rows = builder.limit(rcfg.fetch_k).to_list()
        except Exception:
            # Any hybrid/FTS hiccup falls back to plain vector search.
            builder = table.search(qvec)
            if where:
                builder = builder.where(where, prefilter=True)
            rows = builder.limit(rcfg.fetch_k).to_list()

        return [self._row_to_doc(r) for r in rows[:k]]

    # -- internals --------------------------------------------------------
    def _ensure_fts(self, force: bool = False) -> bool:
        """Ensure a full-text index on ``text`` exists. ``force`` rebuilds it
        (after new data); otherwise it's created only when missing."""
        if self._table_name not in self._db.table_names():
            return False
        table = self._db.open_table(self._table_name)
        try:
            if not force:
                for ix in table.list_indices():
                    if "text" in (getattr(ix, "columns", None) or []):
                        return True
            table.create_fts_index("text", replace=True, use_tantivy=False)
            return True
        except Exception:
            return False

    def _row_to_doc(self, row: Dict[str, Any]) -> Dict[str, Any]:
        meta = {
            "source": row.get("source", "unknown"),
            "page": row.get("page", 0),
            "level": row.get("level", 0),
            "node_type": row.get("node_type", "leaf"),
            "valid_from": row.get("valid_from"),
            "ingested_at": row.get("ingested_at"),
        }
        try:
            meta.update(json.loads(row.get("meta_json") or "{}"))
        except (TypeError, ValueError):
            pass
        meta["stale"] = self._is_stale(row.get("ingested_at"))
        score = row.get("_relevance_score", row.get("_distance"))
        return {"content": row.get("text", ""), "metadata": meta, "score": score}

    def _is_stale(self, ingested_at: Optional[float]) -> bool:
        tcfg = self.settings.temporal
        if not tcfg.enabled or not ingested_at:
            return False
        age_days = (time.time() - float(ingested_at)) / _DAY_SECONDS
        return age_days > tcfg.stale_after_days
