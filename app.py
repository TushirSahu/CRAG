import networkx as nx
import pickle
import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.agent.semantic_cache import LocalSemanticCache  # Your custom cache implementation
import shutil
from src.agent.graph import app as crag_app, vector_db  # Imports your compiled LangGraph and db
from llama_parse import LlamaParse
from langchain_core.documents import Document
import asyncio

# --- 1. Page Setup ---
st.set_page_config(page_title="CRAG Agent Demo", page_icon="🤖", layout="wide")
st.title("🔍 Corrective RAG (CRAG) Agent")

# Check Telemetry status
langsmith_active = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
if langsmith_active:
    st.caption("🟢 **LangSmith Observability is Active:** Full payload traces, cost tracking, and latency metrics are being logged in production.")
else:
    st.caption("🔴 **Telemetry Disabled:** Add your `LANGCHAIN_API_KEY` to the `.env` to enable enterprise-grade observability and cost tracking.")

# --- 2. Sidebar: Document Upload & Ingestion ---
with st.sidebar:
    st.header("📂 Knowledge Base")
    st.markdown("Upload a PDF to teach the agent new information.")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Ingest Document"):
            with st.spinner("Chunking and Embedding Document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # OVERRIDE the temporary file path with the actual filename uploaded
                llama_key = os.getenv("LLAMA_CLOUD_API_KEY")
                if llama_key:
                    # Force standard asyncio event loop to prevent uvloop conflicts with LlamaParse's internal nest_asyncio
                    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
                    
                    st.info("🧠 LlamaParse Vision-Language extraction active (great for tables/images!)")
                    parser = LlamaParse(
                        api_key=llama_key,
                        result_type="markdown",
                        verbose=True
                    )
                    parsed_docs = parser.load_data(tmp_file_path)
                    docs = [Document(page_content=doc.text, metadata={"source": uploaded_file.name}) for doc in parsed_docs]
                else:
                    st.warning("Using legacy PyPDFLoader. Add a free LLAMA_CLOUD_API_KEY to your .env to extract tables and messy PDFs perfectly into Markdown.", icon="⚠️")
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)
                
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Add the new chunks to our existing ChromaDB
                vector_db.add_documents(documents=chunks)
                
                graph_path = "./data/graph_db.pkl"
                if os.path.exists(graph_path):
                    with open(graph_path, 'rb') as f:
                        knowledge_graph = pickle.load(f)
                else:
                    knowledge_graph = nx.Graph()
                    
                knowledge_graph.add_node(uploaded_file.name, type="Document")
                for chunk in chunks[:10]: 
                    entity = chunk.page_content[:25].strip()
                    knowledge_graph.add_node(entity, type="Entity")
                    knowledge_graph.add_edge(uploaded_file.name, entity, relation="CONTAINS")
                    
                with open(graph_path, 'wb') as f:
                    pickle.dump(knowledge_graph, f)
                
                # Clean up the temporary file
                os.remove(tmp_file_path)
                st.success(f"Successfully learned {len(chunks)} chunks from {uploaded_file.name}!")
    st.divider() # Adds a nice visual line
    st.header("🧹 System Controls")
    
    if st.button("Clear Semantic Cache"):
        # This deletes the cache folder from your hard drive
        if os.path.exists("./data/cache_db"):
            shutil.rmtree("./data/cache_db")
            st.session_state.semantic_cache = LocalSemanticCache() # Re-initialize
            st.success("Cache successfully wiped!")
        else:
            st.info("Cache is already empty.")
            
    if st.button("Clear Vector DB (Forget PDFs)"):
        if os.path.exists("./data/chroma_db"):
            shutil.rmtree("./data/chroma_db")
            if os.path.exists("./data/graph_db.pkl"):
                os.remove("./data/graph_db.pkl")
            st.success("Vector DB wiped! The agent has forgotten all PDFs.")
        else:
            st.info("Vector DB is already empty.")

# --- 3. Chat UI ---
st.markdown("""
This agent doesn't just guess. It reads local documents, **grades its own retrieval**, 
and automatically falls back to live web search if its local context is insufficient.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "semantic_cache" not in st.session_state:
    st.session_state.semantic_cache = LocalSemanticCache()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document (or anything else)..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
            # 2. CHECK CACHE FIRST!
            cached_answer = st.session_state.semantic_cache.check_cache(prompt)
            
            if cached_answer:
                # CACHE HIT: Skip the graph entirely! ⚡
                st.success("⚡ Answer retrieved from Semantic Cache (0 latency, $0 cost!)")
                st.markdown(cached_answer)
                st.session_state.messages.append({"role": "assistant", "content": cached_answer})
                
            else:
                # CACHE MISS: Run the full LangGraph agent
                with st.status("Agent is thinking...", expanded=True) as status:
                    # Set initial retry_count to 0 and pass history
                    inputs = {
                        "question": prompt, 
                        "retry_count": 0,
                        "chat_history": st.session_state.messages
                    }
                    final_generation = ""
                    final_documents = []
                    
                    for output in crag_app.stream(inputs):
                        for key, value in output.items():
                            if key == "retrieve":
                                st.write("📚 Retrieving chunks from local DB...")
                            elif key == "grade_documents":
                                st.write("⚖️ Grading documents...")
                            elif key == "web_search":
                                st.write("🌐 Searching the web...")
                            elif key == "rewrite_query":
                                # Show the user that the AI is self-correcting!
                                st.write(f"🔄 Poor results. Rewriting query to: '{value['question']}'")
                            elif key == "generate":
                                st.write("✍️ Synthesizing final answer...")
                                final_generation = value["generation"]
                                final_documents = value.get("documents", [])
                    
                    status.update(label="Process Complete!", state="complete", expanded=False)
                
                st.markdown(final_generation)
                
                # Show sources if any were used
                if final_documents:
                    with st.expander("Show Sources"):
                        for i, doc in enumerate(final_documents):
                            source = doc.get("metadata", {}).get("source", "Unknown Source")
                            page = doc.get("metadata", {}).get("page", "")
                            page_text = f" (Page {page})" if page else ""
                            st.markdown(f"**[Source {i+1}] {source}{page_text}**")
                            st.caption(f"{doc.get('content', '')[:300]}...")

                # 3. SAVE TO CACHE FOR NEXT TIME
                st.session_state.semantic_cache.add_to_cache(prompt, final_generation)
                
                st.session_state.messages.append({"role": "assistant", "content": final_generation})