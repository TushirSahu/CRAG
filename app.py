"""Streamlit UI for the Agentic Knowledge Engine.

Multi-resolution retrieval (RAPTOR + LanceDB hybrid), time-aware evidence, and a
verifiable trust layer that shows a confidence score and abstains when grounding
is weak.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

from src.agent.graph import app as crag_app
from src.agent.nodes import get_cache, get_store
from src.data.ingestion import extract_documents_from_pdf
from src.index.builder import index_documents

st.set_page_config(page_title="Agentic Knowledge Engine", page_icon="🧠", layout="wide")
st.title("🧠 Agentic Knowledge Engine")

langsmith_active = (
    os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    or os.getenv("LANGSMITH_TRACING", "").lower() == "true"
)
st.caption(
    "🟢 LangSmith observability active" if langsmith_active else "🔴 Telemetry disabled (set LANGCHAIN_API_KEY)"
)

# --- Sidebar: ingestion + ops -------------------------------------------
with st.sidebar:
    st.header("📂 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    if uploaded_file is not None and st.button("Ingest Document"):
        with st.spinner("Extracting, building RAPTOR tree, embedding..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                docs = extract_documents_from_pdf(tmp_path, source_name=uploaded_file.name)
                n_nodes = index_documents(docs)
                st.success(f"Indexed {n_nodes} nodes (leaves + summaries) from {uploaded_file.name}!")
            finally:
                os.remove(tmp_path)

    st.divider()
    st.header("⚙️ Status")
    col1, col2 = st.columns(2)
    col1.metric("LangSmith", "Active" if langsmith_active else "Offline")
    try:
        col2.metric("Knowledge Nodes", get_store().count())
    except Exception:
        col2.metric("Knowledge Nodes", 0)

    st.divider()
    if st.button("Clear Semantic Cache"):
        get_cache().clear()
        get_cache.cache_clear()
        st.success("Cache wiped!")
    if st.button("Clear Knowledge Base"):
        get_store.cache_clear()
        get_store().reset()
        get_cache().clear()  # stale answers must not outlive the documents they cited
        get_cache.cache_clear()
        st.success("Knowledge base wiped! The agent has forgotten all documents.")

# --- Chat ----------------------------------------------------------------
st.markdown(
    "Ask a question. The agent retrieves at multiple resolutions, grades its own "
    "evidence, falls back to web search when needed, and reports how confident it is."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents (or anything else)..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.status("Agent is thinking...", expanded=True) as status:
            inputs = {"question": prompt, "retry_count": 0, "chat_history": st.session_state.messages}
            final_generation, final_documents, confidence = "", [], None
            cached, verified = False, True
            labels = {
                "cache_lookup": "⚡ Checking semantic cache...",
                "retrieve": "📚 Retrieving (multi-resolution)...",
                "grade_documents": "⚖️ Grading evidence...",
                "web_search": "🌐 Searching the web...",
                "sql_tool": "🗄️ Querying analytics database...",
                "generate": "✍️ Synthesizing answer...",
                "verify": "🔎 Verifying grounding...",
                "cache_write": "💾 Caching trusted answer...",
                "abstain": "🤖 Abstaining (insufficient evidence)...",
                "unverified": "⚠️ Answer could not be verified...",
            }
            for output in crag_app.stream(inputs):
                for key, value in output.items():
                    if key in labels:
                        st.write(labels[key])
                    if key == "rewrite_query":
                        st.write(f"🔄 Rewriting query to: '{value['question']}'")
                    if value.get("cached"):
                        cached = True
                    if "generation" in value:
                        final_generation = value["generation"]
                    if value.get("documents"):
                        final_documents = value["documents"]
                    if value.get("confidence") is not None:
                        confidence = value["confidence"]
                    if value.get("verified") is not None:
                        verified = value["verified"]
            status.update(label="Process Complete!", state="complete", expanded=False)

        st.markdown(final_generation)
        if cached:
            st.success("⚡ Answer retrieved from the semantic cache.")
        elif not verified:
            st.warning("Could not verify this answer against the sources — treat with caution.")
        elif confidence is not None:
            pct = int(confidence * 100)
            (st.success if pct >= 70 else st.warning if pct >= 50 else st.error)(
                f"Trust score: {pct}% of claims grounded in retrieved evidence."
            )

        if final_documents:
            with st.expander("Show Sources"):
                for i, doc in enumerate(final_documents, start=1):
                    meta = doc.get("metadata", {})
                    stale = " ⏳ stale" if meta.get("stale") else ""
                    level = meta.get("level", 0)
                    kind = "summary" if meta.get("node_type") == "summary" else "leaf"
                    st.markdown(f"**[Source {i}] {meta.get('source', 'Unknown')}** · {kind} L{level}{stale}")
                    st.caption(f"{doc.get('content', '')[:300]}...")

        st.session_state.messages.append({"role": "assistant", "content": final_generation})
