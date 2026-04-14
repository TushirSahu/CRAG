# 🔍 Agentic CRAG (Corrective RAG) System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green)](https://github.com/langchain-ai/langchain)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange)](https://deepmind.google/technologies/gemini/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)

A production-ready **Corrective Retrieval-Augmented Generation (CRAG)** application built with **LangGraph** and **Streamlit**. 

Unlike standard RAG pipelines that blindly rely on retrieved context—even when it's poor or irrelevant—this agentic workflow **self-evaluates** the quality of its retrieval. If the local document context is insufficient, it autonomously falls back to a live web search (via Tavily) and iteratively rewrites queries to find the correct information.

## ✨ Key Features

- **Advanced Agentic Orchestration (LangGraph):** State-machine driven workflow enforcing robust control logic (Retrieve ➡️ Grade Docs ➡️ Generate ➡️ Grade Answer ➡️ Web Fallback/End).
- **Zero-Hallucination Guardrails (Self-RAG):** Evaluates generated answers against retrieved documents. Ungrounded (hallucinated) answers are instantly rejected, triggering an autonomous query rewrite and web search.
- **Document Relevance Grading:** Uses an LLM Grader to score retrieval document relevance, filtering out noisy context before generation.
- **Embedding Fine-Tuning (Domain Adaptation):** Pipeline to generate synthetic QA datasets and fine-tune `HuggingFaceEmbeddings` via Contrastive Learning (Multiple Negatives Ranking Loss) for high domain precision.
- **Ragas Evaluation Suite:** Automated testing scripts measuring *Faithfulness, Answer Relevancy, Context Precision,* and *Context Recall*.
- **Dynamic Query Rewriting:** If web search or local context fails, the agent rewrites its own search queries to extract better results.
- **Semantic Caching:** Zero-latency retrieval for previously asked semantic queries, severely slashing API costs.
- **Verifiable Output:** Generations provide **clean inline citations**, explicitly referencing source PDF filenames and page numbers.
- **Observable & Deployable:** Monitored with **LangSmith Tracing** and fully containerized with **Docker**.

## 🧠 System Architecture

```mermaid
graph TD
    A[User Query] --> B(Retrieve Documents from ChromaDB)
    B --> C{LLM Grader: Are Docs Relevant?}
    
    C -- Yes --> D((Generate LLM Answer))
    C -- No / Partial --> E(Web Search via Tavily)
    
    E --> F{Check Search Results}
    F -- Sufficient --> D
    F -- Insufficient --> G(Rewrite Query)
    G --> E
    
    D --> I{Hallucination Grader: Is Answer Grounded?}
    I -- Yes --> H[Final Evaluated Response + Sources]
    I -- No --> G
```

```text
CRAG/
├── src/                    # Source code
│   ├── agent/              # LangGraph Agent Logic
│   │   ├── __init__.py
│   │   ├── graph.py        # Core orchestration (CRAG + Self-RAG)
│   │   └── semantic_cache.py # Semantic caching logic
│   └── utils/              # Utility & ML scripts
│       ├── __init__.py
│       ├── finetune_embeddings.py # Contrastive Learning domain adaptation
│       ├── generate_synthetic_data.py # Synthetic QA dataset generation
│       └── ingest.py
├── data/                   # Local databases (git-ignored)
│   ├── chroma_db/          # Persistent Vector Store
│   ├── cache_db/           # Persistent Cache Store
│   └── finetuned-domain-embeddings/ # Custom local embedding model
├── tests/                  # Evaluation and testing
│   ├── __init__.py
│   └── evaluate_rag.py     # Ragas evaluation pipeline
├── app.py                  # Streamlit Frontend application
├── Dockerfile              # Containerization
├── requirements.txt        # Dependencies
└── .env                    # Environment variables (Google API, Tavily, LangSmith)
```

## 🛠 Tech Stack

- **Orchestration:** LangGraph, LangChain
- **LLM:** Google Gemini (`gemini-2.5-flash`)
- **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Database:** ChromaDB (Local SQLite)
- **Web Search API:** Tavily
- **Frontend UI:** Streamlit
- **DevOps:** Docker, LangSmith

## 🚀 Getting Started

### Prerequisites
Make sure you have an API key for **Google Gemini** and **Tavily**. 

### 1. Bare-metal Installation
```bash
# Clone the repository
git clone https://github.com/your-username/CRAG.git
cd CRAG

# Install dependencies
pip install -r requirements.txt

# Set your environment variables
echo "GOOGLE_API_KEY=your_gemini_key" >> .env
echo "TAVILY_API_KEY=your_tavily_key" >> .env
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_API_KEY=your_langsmith_key" >> .env

# Run the app
streamlit run app.py
```

### 2. Docker Deployment
```bash
# Build the image
docker build -t crag-agent .

# Run the container (mapping ports and passing local .env)
docker run -p 8501:8501 --env-file .env crag-agent
```

## 💡 How to Use

1. **Upload Documents:** Use the sidebar to upload any PDF. It will be immediately chunked and embedded into the local ChromaDB.
2. **Chat with the Agent:** Ask highly specific questions about the document. 
3. **Observe the Agent:** The Streamlit UI will display the LangGraph state transitions in real time (`Retrieving...`, `Grading...`, `Web Searching...`, etc.).
4. **Verify Sources:** Check the `### References` generated at the bottom of responses, or expand the "Show Sources" accordion to view the raw snippet chunks used.

## 🔜 Future Roadmap

- Integrate **LangGraph Checkpointers** for robust, cross-session thread database persistence.
- Implement **Token Streaming** to the frontend UI to eliminate perceived generation latency.
- Displace API dependencies by fine-tuning a **Small Language Model (e.g. LLaMA-3 8B via LoRA)** to locally execute binary document/hallucination grading.
- Support multimodal document ingestion (images, charts, tables).