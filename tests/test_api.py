"""Smoke test for the CRAG API handler.

The test stubs the LangGraph app so the handler can be exercised without
calling the live Gemini-backed pipeline or requiring FastAPI at runtime.
"""
import sys
import os
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import api.main as api_main


class _FakeCRAGApp:
	def stream(self, inputs):  # noqa: D401 - test double
		yield {"generate": {"generation": f"mock answer for: {inputs['question']}", "documents": []}}


def _run_smoke_check():
	with patch.object(api_main, "get_crag_app", lambda: _FakeCRAGApp()):
		response = api_main.chat_with_agent(api_main.QueryRequest(question="What is ASC 842?"))

	assert "response" in response
	assert response["response"].startswith("mock answer")
	return response


def test_chat_endpoint_smoke():
	_run_smoke_check()


if __name__ == "__main__":
	_run_smoke_check()
	print("Kind of test successful")
