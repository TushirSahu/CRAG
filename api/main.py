from __future__ import annotations
import os
import sys

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - keeps the module importable in lightweight test environments
    class BaseModel:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

    class _FallbackFastAPI:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def _route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        post = _route
        get = _route

    FastAPI = _FallbackFastAPI

# Add root directory to python path to import existing CRAG logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


app = FastAPI(title="Agentic Knowledge Engine API", description="🧠 Trust-aware agentic RAG", version="2.0")

class QueryRequest(BaseModel):
    question: str


def _agent_run(question: str, **kwargs):
    """Indirection so the heavy LangGraph import stays lazy (and patchable in tests)."""
    from src.agent.graph import run

    return run(question, **kwargs)

@app.post("/chat")
def chat_with_agent(request: QueryRequest):
    """Run the agent for one question and return the answer with its trust score."""
    result = _agent_run(request.question, retry_count=0, chat_history=[])
    return {
        "response": result.get("generation", ""),
        "confidence": result.get("confidence"),
        "sources": [d.get("metadata", {}).get("source") for d in result.get("documents", [])],
    }


@app.get("/")
def read_root():
    """Simple root endpoint for health and docs link."""
    return {"status": "ok", "message": "CRAG Agent API running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)