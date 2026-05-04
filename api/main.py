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

        def post(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    FastAPI = _FallbackFastAPI

# Add root directory to python path to import existing CRAG logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_crag_app():
    from src.agent.graph import app as crag_app

    return crag_app

app = FastAPI(title="CRAG Agent API", description="🚀 Corrective RAG Backend API", version="1.0")

class QueryRequest(BaseModel):
    question: str

@app.post("/chat")
def chat_with_agent(request: QueryRequest):
    """
    Exposes the LangGraph Agent as a REST API endpoint.
    """
    inputs = {"question": request.question, "retry_count": 0, "chat_history": []}
    final_generation = ""

    crag_app = get_crag_app()
    
    for output in crag_app.stream(inputs):
        for key, value in output.items():
            if key == "generate":
                final_generation = value["generation"]
                
    return {"response": final_generation}


@app.get("/")
def read_root():
    """Simple root endpoint for health and docs link."""
    return {"status": "ok", "message": "CRAG Agent API running", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)