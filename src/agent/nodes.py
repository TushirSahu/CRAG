"""Agent node functions and routing decisions.

Lean, config-driven replacements for the logic that used to live inline in the
417-line ``graph.py``. Retrieval now goes through :class:`KnowledgeStore`
(LanceDB hybrid) — the old BM25/RRF ensemble is gone. Generation is followed by
a verifiable trust check (Pillar B) and retrieval is temporally aware (Pillar A).
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import TYPE_CHECKING, List, TypedDict

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

from src.agent import prompts
from src.agent.trust import verify_generation
from src.core.config import get_settings
from src.core.embeddings import get_embeddings
from src.index.vectorstore import KnowledgeStore
from src.utils.logger import logger

if TYPE_CHECKING:
    from src.agent.sql_tool import SQLTool


class GraphState(TypedDict, total=False):
    question: str
    original_question: str  # the user's phrasing, preserved across rewrites (cache key)
    cached: bool            # answer served from the semantic cache
    chat_history: List[dict]
    generation: str
    web_search_needed: bool
    documents: List[dict]
    retry_count: int
    as_of: float          # Pillar A: point-in-time queries (epoch seconds)
    sources: List[str]    # source ACL filter
    confidence: float     # Pillar B: trust score of the final answer
    verified: bool        # whether the trust check itself ran successfully
    trust: dict


@lru_cache(maxsize=4)
def get_llm(role: str = "generate"):
    from langchain_google_genai import ChatGoogleGenerativeAI

    cfg = get_settings().models
    model = cfg.llm if role == "generate" else cfg.grader_llm
    temperature = cfg.generation_temperature if role == "generate" else 0
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


@lru_cache(maxsize=1)
def get_store() -> KnowledgeStore:
    return KnowledgeStore(get_embeddings())


@lru_cache(maxsize=1)
def get_cache():
    from src.agent.semantic_cache import LocalSemanticCache

    return LocalSemanticCache()


# --- nodes ---------------------------------------------------------------
def cache_lookup(state: GraphState) -> GraphState:
    """Entry node: short-circuit to a cached answer for a semantically similar question."""
    logger.info("--- Cache lookup ---")
    question = state["question"]
    answer = get_cache().check_cache(question)
    if answer is not None:
        return {"generation": answer, "cached": True, "original_question": question, "documents": []}
    return {"cached": False, "original_question": question}


def cache_write(state: GraphState) -> GraphState:
    """Persist a trusted answer under the user's original question phrasing."""
    logger.info("--- Cache write ---")
    get_cache().add_to_cache(state.get("original_question", state["question"]), state["generation"])
    return {}


def retrieve(state: GraphState) -> GraphState:
    logger.info("--- Retrieve (LanceDB hybrid + temporal) ---")
    docs = get_store().search(
        state["question"],
        as_of=state.get("as_of"),
        sources=state.get("sources"),
    )
    return {"documents": docs, "question": state["question"]}


def grade_documents(state: GraphState) -> GraphState:
    logger.info("--- Grade documents ---")
    grader = prompts.GRADE_DOCUMENTS | get_llm("grade") | StrOutputParser()
    pause = get_settings().agent.grade_rate_limit_seconds
    relevant = []
    for doc in state["documents"]:
        score = grader.invoke({"question": state["question"], "document": doc["content"]})
        if "yes" in score.strip().lower():
            relevant.append(doc)
        if pause:
            time.sleep(pause)
    return {
        "documents": relevant,
        "web_search_needed": not relevant,
        "question": state["question"],
    }


def web_search(state: GraphState) -> GraphState:
    logger.info("--- Web search fallback ---")
    documents = state.get("documents", []) if state.get("retry_count", 0) == 0 else []
    results = TavilySearchResults(max_results=2).invoke({"query": state["question"]})
    if isinstance(results, list):
        for d in results:
            if isinstance(d, dict) and "content" in d:
                documents.append({"content": d["content"], "metadata": {"source": d.get("url", "Web Search")}})
            else:
                documents.append({"content": str(d), "metadata": {"source": "Web Search"}})
    else:
        documents.append({"content": str(results), "metadata": {"source": "Web Search"}})
    return {"documents": documents, "question": state["question"]}


@lru_cache(maxsize=1)
def get_sql_tool() -> "SQLTool":
    from src.agent.sql_tool import SQLTool

    return SQLTool(llm=get_llm("grade"))


def run_sql(state: GraphState) -> GraphState:
    logger.info("--- SQL tool (read-only text-to-SQL) ---")
    tool = get_sql_tool()
    if not tool.available:
        doc = {
            "content": "No analytics database is configured. Run scripts/seed_analytics_db.py.",
            "metadata": {"source": "SQL Database"},
        }
        return {"documents": [doc], "question": state["question"]}
    try:
        doc = tool.query(state["question"])
    except Exception as exc:  # generation/validation/execution failure → surface, don't crash
        logger.warning("SQL tool failed: %s", exc)
        doc = {
            "content": f"The analytics query could not be completed: {exc}",
            "metadata": {"source": "SQL Database"},
        }
    return {"documents": [doc], "question": state["question"]}


def generate(state: GraphState) -> GraphState:
    logger.info("--- Generate ---")
    history = state.get("chat_history", [])
    history_text = ""
    if history:
        history_text = "Previous Conversation:\n" + "".join(
            f"{m['role'].capitalize()}: {m['content']}\n" for m in history[-4:]
        )

    blocks = []
    for i, doc in enumerate(state["documents"], start=1):
        meta = doc["metadata"]
        stale = " (stale)" if meta.get("stale") else ""
        tag = f"[Source {i}] ({meta.get('source', 'Unknown')}, Page {meta.get('page', 'N/A')}){stale}"
        blocks.append(f"{tag}:\n{doc['content']}")

    chain = prompts.GENERATE | get_llm("generate") | StrOutputParser()
    generation = chain.invoke(
        {"context": "\n\n".join(blocks), "question": state["question"], "history_text": history_text}
    )
    return {"generation": generation, "documents": state["documents"], "question": state["question"]}


def verify(state: GraphState) -> GraphState:
    """Pillar B: score how well the answer is grounded in the retrieved context."""
    logger.info("--- Verify (trust layer) ---")
    report = verify_generation(state["generation"], state["documents"], get_llm("grade"))
    logger.info(
        "confidence=%.2f (%d/%d claims) verified=%s",
        report["confidence"], report["supported"], report["total"], report["verified"],
    )
    return {"confidence": report["confidence"], "verified": report["verified"], "trust": report}


def abstain(state: GraphState) -> GraphState:
    logger.info("--- Abstain (insufficient grounded evidence) ---")
    cfg = get_settings().trust
    return {"generation": cfg.abstain_message, "confidence": state.get("confidence", 0.0)}


def unverified(state: GraphState) -> GraphState:
    """The verifier couldn't run — return the answer, but flagged as unverified."""
    logger.info("--- Unverified (trust check unavailable) ---")
    cfg = get_settings().trust
    return {"generation": cfg.unverified_prefix + state.get("generation", "")}


def rewrite_query(state: GraphState) -> GraphState:
    logger.info("--- Rewrite query ---")

    class Rewritten(BaseModel):
        optimized_query: str = Field(description="Optimized search query. No conversational text.")

    parser = JsonOutputParser(pydantic_object=Rewritten)
    chain = prompts.REWRITE_QUERY | get_llm("grade") | parser
    result = chain.invoke(
        {"question": state["question"], "format_instructions": parser.get_format_instructions()}
    )
    return {"question": result["optimized_query"], "retry_count": state.get("retry_count", 0) + 1}


# --- routing decisions ---------------------------------------------------
def decide_after_cache(state: GraphState) -> str:
    """Skip the whole pipeline on a cache hit; otherwise route by intent."""
    if state.get("cached"):
        return "cached"
    return route_query_intent(state)


def route_query_intent(state: GraphState) -> str:
    decision = (prompts.ROUTER | get_llm("grade") | StrOutputParser()).invoke(
        {"question": state["question"]}
    ).strip().lower()
    if "web" in decision:
        return "web_search"
    if "sql" in decision:
        return "sql_tool"
    return "retrieve"


def decide_after_grade(state: GraphState) -> str:
    return "web_search" if state.get("web_search_needed") else "generate"


def decide_after_web_search(state: GraphState) -> str:
    return "generate" if state.get("retry_count", 0) >= get_settings().agent.max_retries else "rewrite_query"


def decide_after_verify(state: GraphState) -> str:
    """Accept, retry, abstain, or flag-unverified based on the trust check.

    A *verified* but poorly-grounded answer abstains; an answer the verifier
    *couldn't check* is returned with an unverified warning instead — abstaining
    there would discard a possibly-good answer over a checker failure.
    """
    cfg = get_settings()
    if state.get("verified", True) and state.get("confidence", 0.0) >= cfg.trust.min_confidence:
        return "useful"
    if state.get("retry_count", 0) < cfg.agent.max_retries:
        return "retry"
    return "abstain" if state.get("verified", True) else "unverified"
