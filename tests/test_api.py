"""Smoke test for the API handler.

Stubs the agent's ``run`` helper so the handler can be exercised without the
live Gemini pipeline or API keys.
"""
import os
import sys
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api.main as api_main


def _fake_run(question, **kwargs):
    return {
        "generation": f"mock answer for: {question}",
        "confidence": 0.9,
        "documents": [{"metadata": {"source": "doc.pdf"}}],
    }


def _run_smoke_check():
    with patch.object(api_main, "_agent_run", _fake_run):
        response = api_main.chat_with_agent(api_main.QueryRequest(question="What is ASC 842?"))

    assert response["response"].startswith("mock answer")
    assert response["confidence"] == 0.9
    assert response["sources"] == ["doc.pdf"]
    return response


def test_chat_endpoint_smoke():
    _run_smoke_check()


if __name__ == "__main__":
    _run_smoke_check()
    print("✅ api smoke test passed")
