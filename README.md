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
- **Mathematical Evaluation:** Automated pipeline to prove system effectiveness mathematically using the **RAGAS** framework (Faithfulness, Answer Relevancy, Context Precision, Context Recall).
- **Web Fallback:** Autonomous Tavily web search for out-of-domain questions or rejected context.

## 🧠 System Flow

1. **Retrieve:** Hybrid Search (BM25 + Chroma) + GraphRAG network contexts.
2. **Grade Docs:** LLM Grader assesses if the retrieved chunks actually answer the query.
3. **Web Fallback:** If docs are irrelevant, query is rewritten and passed to Tavily web search.
4. **Generate & Check Hallucinations:** Generates answer, then evaluates if the response is safely grounded in the provided context before showing it to the user.

## 🚀 Getting Started

**1. Install Dependencies**
```bash
git clone https://github.com/your-username/CRAG.git
cd CRAG
pip install -r requirements.txt
```

**2. Configure Environment (`.env`)**
```env
GOOGLE_API_KEY=your_gemini_key
TAVILY_API_KEY=your_tavily_key
LLAMA_CLOUD_API_KEY=your_llamaparse_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
```

**3. Run the App**
```bash
streamlit run app.py
```

**4. Run with Docker / Docker Compose**
Using Docker:
```bash
docker build -t crag-agent .
docker run -p 8501:8501 --env-file .env crag-agent
```

Or simply using Docker Compose:
```bash
docker-compose up --build -d
```

**# App UI runs at http://localhost:8501**
