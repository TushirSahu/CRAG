"""RAPTOR tree-building test with a fake summarizer/embedder (no LLM/API keys)."""

import numpy as np

from src.index.raptor import build_raptor_nodes


class ClusterableEmbedder:
    """Embeds text into 3 well-separated regions based on a keyword, so GMM
    produces meaningful clusters deterministically."""

    dim = 16
    groups = {"finance": 0, "biology": 5, "sports": 10}

    def _vec(self, text):
        v = np.zeros(self.dim, dtype=np.float32)
        base = next((o for kw, o in self.groups.items() if kw in text.lower()), 0)
        v[base] = 1.0
        v[base + 1] = 0.5
        # tiny deterministic jitter so members of a cluster aren't identical
        v[base + 2] = (len(text) % 5) * 0.01
        return v.tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


def fake_summarize(text):
    return f"SUMMARY[{len(text.split())} words]: {text[:40]}"


def test_raptor_builds_multilevel_tree():
    leaves = [
        {"content": f"finance report quarter {i} revenue and profit", "metadata": {"source": "fin.pdf"}}
        for i in range(4)
    ] + [
        {"content": f"biology notes on cells and genome part {i}", "metadata": {"source": "bio.pdf"}}
        for i in range(4)
    ] + [
        {"content": f"sports match results and scores game {i}", "metadata": {"source": "spt.pdf"}}
        for i in range(4)
    ]

    nodes = build_raptor_nodes(leaves, ClusterableEmbedder(), fake_summarize)

    leaf_nodes = [n for n in nodes if n["metadata"]["node_type"] == "leaf"]
    summary_nodes = [n for n in nodes if n["metadata"]["node_type"] == "summary"]

    assert len(leaf_nodes) == 12, "all leaves preserved"
    assert summary_nodes, "at least one summary node produced"
    assert all("vector" in n for n in nodes), "every node carries an embedding"
    assert all(n["metadata"]["level"] >= 1 for n in summary_nodes)
    assert any("SUMMARY[" in n["content"] for n in summary_nodes)


if __name__ == "__main__":
    test_raptor_builds_multilevel_tree()
    print("✅ raptor test passed")
