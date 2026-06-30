"""Tests for the semantic cache and its routing — no API keys needed.

Uses the deterministic ``HashEmbedder`` from the vectorstore tests so identical
questions produce identical vectors (cosine similarity 1.0).
"""

from src.agent import nodes
from src.agent.semantic_cache import LocalSemanticCache
from tests.test_vectorstore import HashEmbedder


def _cache(tmp_path):
    return LocalSemanticCache(
        embedder=HashEmbedder(), threshold=0.9, path=str(tmp_path / "cache")
    )


def test_miss_then_hit(tmp_path):
    cache = _cache(tmp_path)
    assert cache.check_cache("how many signups yesterday") is None
    cache.add_to_cache("how many signups yesterday", "There were 47.")
    assert cache.check_cache("how many signups yesterday") == "There were 47."


def test_clear_invalidates(tmp_path):
    cache = _cache(tmp_path)
    cache.add_to_cache("q", "a")
    cache.clear()
    assert cache.check_cache("q") is None


def test_decide_after_cache_short_circuits():
    assert nodes.decide_after_cache({"cached": True}) == "cached"
    # On a miss it delegates to intent routing — covered there, not re-run here.


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        test_miss_then_hit(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_clear_invalidates(Path(d))
    test_decide_after_cache_short_circuits()
    print("✅ cache tests passed")
