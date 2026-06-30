"""RAPTOR hierarchical indexing.

Instead of a flat list of chunks, RAPTOR recursively **clusters** chunks and
**summarizes** each cluster with the LLM, producing a tree: raw leaves at the
bottom, progressively higher-level summaries above. Indexing every level gives
*multi-resolution* retrieval — detail questions match leaves, "big picture"
questions match summaries.

Clustering uses a Gaussian Mixture Model (component count chosen by BIC) on
optionally PCA-reduced embeddings. The summarizer is injected as a callable so
this module is LLM-agnostic and unit-testable without API keys.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.core.config import Settings, get_settings
from src.core.embeddings import Embedder

Summarizer = Callable[[str], str]
Node = Dict[str, Any]  # {"content": str, "metadata": {...}, "vector": list[float]}

_MAX_COMPONENTS = 10


def _choose_n_components(matrix: np.ndarray, max_k: int) -> int:
    """Pick the GMM component count that minimizes BIC (model selection)."""
    best_k, best_bic = 1, float("inf")
    for k in range(1, max_k + 1):
        try:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=k, random_state=42).fit(matrix)
            bic = gmm.bic(matrix)
        except Exception:
            continue
        if bic < best_bic:
            best_k, best_bic = k, bic
    return best_k


def _cluster(matrix: np.ndarray, reduce_dim: int) -> np.ndarray:
    """Return a cluster label per row."""
    from sklearn.mixture import GaussianMixture

    data = matrix
    if reduce_dim and matrix.shape[1] > reduce_dim and matrix.shape[0] > reduce_dim:
        from sklearn.decomposition import PCA

        data = PCA(n_components=reduce_dim, random_state=42).fit_transform(matrix)

    max_k = min(_MAX_COMPONENTS, data.shape[0] - 1)
    n = _choose_n_components(data, max_k)
    return GaussianMixture(n_components=n, random_state=42).fit_predict(data)


def build_raptor_nodes(
    leaves: Sequence[Node],
    embedder: Embedder,
    summarize: Summarizer,
    settings: Optional[Settings] = None,
) -> List[Node]:
    """Build leaves + recursive summary nodes ready for indexing.

    Each input leaf is ``{"content": str, "metadata": {...}}``; returned nodes
    additionally carry ``vector`` and ``metadata['level']`` / ``['node_type']``.
    """
    cfg = (settings or get_settings()).raptor
    leaves = list(leaves)
    if not leaves:
        return []

    # Embed and tag the base level.
    base_vectors = embedder.embed_documents([n["content"] for n in leaves])
    all_nodes: List[Node] = []
    for node, vec in zip(leaves, base_vectors):
        meta = dict(node.get("metadata", {}))
        meta.setdefault("level", 0)
        meta.setdefault("node_type", "leaf")
        all_nodes.append({"content": node["content"], "metadata": meta, "vector": list(vec)})

    if not cfg.enabled:
        return all_nodes

    # Recursively cluster + summarize upward.
    current = all_nodes
    for level in range(1, cfg.max_levels + 1):
        if len(current) <= cfg.min_cluster_size:
            break
        matrix = np.array([n["vector"] for n in current], dtype=np.float32)
        labels = _cluster(matrix, cfg.reduce_dim)

        summaries: List[Node] = []
        for label in sorted(set(labels)):
            members = [n for n, lab in zip(current, labels) if lab == label]
            if len(members) < 2:
                continue  # nothing to summarize from a singleton
            joined = "\n\n".join(m["content"] for m in members)
            summary_text = summarize(joined)
            sources = sorted({str(m["metadata"].get("source", "unknown")) for m in members})
            summaries.append(
                {
                    "content": summary_text,
                    "metadata": {
                        "level": level,
                        "node_type": "summary",
                        "source": ", ".join(sources),
                        "n_children": len(members),
                    },
                }
            )

        if not summaries:
            break
        sum_vectors = embedder.embed_documents([s["content"] for s in summaries])
        for s, vec in zip(summaries, sum_vectors):
            s["vector"] = list(vec)
        all_nodes.extend(summaries)
        current = summaries

    return all_nodes
