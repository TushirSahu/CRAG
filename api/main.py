from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Add root directory to python path to import existing CRAG logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.agent.graph import app as crag_app

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
    
    for output in crag_app.stream(inputs):
        for key, value in output.items():
            if key == "generate":
                final_generation = value["generation"]
                
    return {"response": final_generation}
