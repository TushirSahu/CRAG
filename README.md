# 🔍 Enterprise Agentic CRAG (Corrective RAG)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)

A production-ready **Corrective Retrieval-Augmented Generation (CRAG)** architecture built with **LangGraph** and **Streamlit**. It self-evaluates retrieval quality, autonomously falling back to web search if local context is insufficient.

## ✨ Key Features

- **Advanced Agentic Orchestration:** State-machine workflow via LangGraph with Self-RAG guardrails (Hallucination & Document grading).
- **Hybrid Search (Custom RRF):** Combines BM25 exact-keyword matching with Chroma DB semantic search using Reciprocal Rank Fusion.
- **GraphRAG Integration:** Extracts entity relationships via NetworkX during ingestion to enrich LLM context dynamically.
- **Vision-Language Ingestion:** Uses LlamaParse to accurately extract nested data and complex financial tables from messy PDFs.
- **Embedding Fine-Tuning:** Pipeline for domain adaptation using synthetic QA generation and Contrastive Learning (Multiple Negatives Ranking Loss) to fine-tune HuggingFace embeddings.
- **Mathematical Evaluation:** Automated pipeline to prove system effectiveness mathematically using the **RAGAS** framework (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
- **Web Fallback:** Autonomous Tavily web search for out-of-domain questions or rejected context.

## 🧠 System Flow

1. **Retrieve:** Hybrid Search (BM25 + Chroma) + GraphRAG network contexts.
2. **Grade Docs:** LLM Grader assesses if the retrieved chunks actually answer the query.
3. **Web Fallback:** If docs are irrelevant, query is rewritten and passed to Tavily web search.
4. **Generate & Check Hallucinations:** Generates answer, then evaluates if the response is safely grounded in the provided context before showing it to the user.

## ⚙️ Enterprise MLOps Architecture

This repository isn't just a script—it's a decoupled **LLMOps System** designed for production inference and continuous data flywheels.

- **FastAPI Microservice:** The core LangGraph agent is isolated behind a scalable REST API.
- **Containerization & IaC:** Dockerized environments, Kubernetes manifests (`k8s/`), and Terraform modules (`terraform/`).
- **CI/CD & DVC:** Automated workflows (`.github/`) and Data Version Control (`dvc.yaml`) for embedding model parameters.
- **Custom Logging:** Resilient global logging and traceback captures (`src/utils`).

## 🚀 Getting Started

**1. Install the Project Package**
```bash
git clone https://github.com/your-username/CRAG.git
cd CRAG
pip install -r requirements.txt
pip install -e .  # Installs the local CRAG package
```

**2. Configure Environment (`.env`)**
```env
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
LLAMA_CLOUD_API_KEY=your_llamaparse_key
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
```

**3. Run the Backend API (FastAPI)**
```bash
uvicorn api.main:app --reload
# Access Interactive Swagger Docs: http://localhost:8000/docs
```

**4. Run the Visual Frontend (Streamlit)**
```bash
streamlit run app.py
# Access Human-in-the-loop Agent UI: http://localhost:8501
```

**5. Run with Docker / Docker Compose**
Using Docker:
```bash
docker build -t crag-agent .
docker run -p 8501:8501 -p 8000:8000 --env-file .env crag-agent
```

Or simply using Docker Compose:
```bash
docker-compose up --build -d
```
