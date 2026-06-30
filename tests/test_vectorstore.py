"""Runtime tests for the LanceDB KnowledgeStore using a deterministic fake embedder.

These run without the LLM stack or API keys — they exercise hybrid search,
temporal (`as_of`) filtering, source ACL, and staleness flags directly.
"""

import time

import numpy as np

from src.core.config import get_settings
from src.index.vectorstore import KnowledgeStore


class HashEmbedder:
    """Cheap deterministic embedder: bag-of-chars hashed into a fixed vector."""

    dim = 64

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            v[hash(token) % self.dim] += 1.0
        norm = np.linalg.norm(v)
        return (v / norm if norm else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def _fresh_store(tmp_path):
    settings = get_settings().model_copy(deep=True)
    settings.vector_store.path = str(tmp_path / "lancedb")
    settings.vector_store.table = "test_kb"
    store = KnowledgeStore(HashEmbedder(), settings=settings)
    store.reset()
    return store


def test_hybrid_search_and_acl(tmp_path):
    store = _fresh_store(tmp_path)
    store.add(
        ["the cat sat on the mat", "dogs are loyal animals", "the mat was blue and soft"],
        [{"source": "a.pdf", "page": 1}, {"source": "b.pdf", "page": 2}, {"source": "a.pdf", "page": 3}],
    )
    assert store.count() == 3

    hits = store.search("mat", k=3)
    assert hits and any("mat" in h["content"] for h in hits)

    # Source ACL: restrict to b.pdf only.
    restricted = store.search("animals", sources=["b.pdf"])
    assert restricted and all(h["metadata"]["source"] == "b.pdf" for h in restricted)


def test_temporal_filter_and_staleness(tmp_path):
    store = _fresh_store(tmp_path)
    now = time.time()
    old = now - 400 * 86_400  # 400 days ago → stale (default 180)
    store.add(
        ["policy v1 says vacation is 10 days", "policy v2 says vacation is 20 days"],
        [
            {"source": "policy", "valid_from": old, "ingested_at": old},
            {"source": "policy", "valid_from": now, "ingested_at": now},
        ],
    )

    # Point-in-time query: as of just after v1, only v1 is valid.
    past = store.search("vacation policy", as_of=old + 1)
    assert past and all("v1" in h["content"] for h in past)
    assert all(h["metadata"]["stale"] for h in past)  # the old doc is flagged stale

    # Current query sees the newest, non-stale fact too.
    current = store.search("vacation policy")
    assert any("v2" in h["content"] and not h["metadata"]["stale"] for h in current)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_hybrid_search_and_acl(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_temporal_filter_and_staleness(Path(d))
    print("✅ vectorstore tests passed")
