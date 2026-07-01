"""LanceDB-backed semantic cache (query → answer) using the shared embedder.

Same idea as before — return a previous answer when a semantically similar
question arrives — but on LanceDB and sharing the single app-wide embedding
model instead of instantiating its own.
"""

from __future__ import annotations

import time
import uuid
from typing import Optional

from src.core import lancedb as ldb
from src.core.config import get_settings
from src.core.embeddings import Embedder, get_embeddings
from src.utils.logger import logger


class LocalSemanticCache:
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        threshold: Optional[float] = None,
        path: Optional[str] = None,
    ):
        cfg = get_settings().cache
        self.embedder = embedder or get_embeddings()
        self.threshold = cfg.threshold if threshold is None else threshold  # min cosine similarity
        self._db = ldb.connect(path or cfg.path)
        self._table = "semantic_cache"

    def check_cache(self, query: str):
        if self._table not in self._db.table_names():
            return None
        results = (
            self._db.open_table(self._table)
            .search(self.embedder.embed_query(query))
            .metric("cosine")
            .limit(1)
            .to_list()
        )
        if results:
            similarity = 1.0 - float(results[0].get("_distance", 1.0))
            if similarity >= self.threshold:
                logger.info("⚡ Cache hit (similarity %.2f)", similarity)
                return results[0]["answer"]
        return None

    def clear(self) -> None:
        if self._table in self._db.table_names():
            self._db.drop_table(self._table)

    def add_to_cache(self, query: str, answer: str) -> None:
        row = {
            "id": str(uuid.uuid4()),
            "query": query,
            "answer": answer,
            "vector": self.embedder.embed_query(query),
            "created_at": time.time(),
        }
        if self._table in self._db.table_names():
            self._db.open_table(self._table).add([row])
        else:
            self._db.create_table(self._table, data=[row])
