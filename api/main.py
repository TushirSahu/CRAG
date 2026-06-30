"""FastAPI surface for the Agentic Knowledge Engine.

One ``/chat`` endpoint runs the LangGraph agent and returns the answer with its
trust signals. The heavy LangGraph import is deferred to request time (via
``_agent_run``) so importing this module — e.g. in tests — stays cheap and the
indirection stays patchable.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Agentic Knowledge Engine API",
    description="🧠 Trust-aware agentic RAG",
    version="2.0",
)


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
        "verified": result.get("verified", True),
        "cached": result.get("cached", False),
        "sources": [d.get("metadata", {}).get("source") for d in result.get("documents", [])],
    }


@app.get("/")
def read_root():
    """Simple root endpoint for health and docs link."""
    return {"status": "ok", "message": "CRAG Agent API running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
