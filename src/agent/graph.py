"""Agentic Knowledge Engine — graph wiring only.

The node logic lives in :mod:`src.agent.nodes`; this file just assembles the
LangGraph state machine:

    route ─┬─ retrieve → grade ─┬─ web_search ─┬─ rewrite → web_search
           │                    └─ generate    └─ generate
           ├─ web_search → (rewrite loop) → generate
           └─ sql_tool → generate
    generate → verify → useful | retry→rewrite | abstain
"""

from __future__ import annotations

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.agent import nodes
from src.agent.nodes import GraphState

load_dotenv()


def build_app():
    g = StateGraph(GraphState)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("web_search", nodes.web_search)
    g.add_node("sql_tool", nodes.run_sql)
    g.add_node("generate", nodes.generate)
    g.add_node("verify", nodes.verify)
    g.add_node("abstain", nodes.abstain)
    g.add_node("rewrite_query", nodes.rewrite_query)

    g.set_conditional_entry_point(
        nodes.route_query_intent,
        {"web_search": "web_search", "sql_tool": "sql_tool", "retrieve": "retrieve"},
    )
    g.add_edge("sql_tool", "generate")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents", nodes.decide_after_grade,
        {"web_search": "web_search", "generate": "generate"},
    )
    g.add_conditional_edges(
        "web_search", nodes.decide_after_web_search,
        {"rewrite_query": "rewrite_query", "generate": "generate"},
    )
    g.add_edge("rewrite_query", "web_search")
    g.add_edge("generate", "verify")
    g.add_conditional_edges(
        "verify", nodes.decide_after_verify,
        {"useful": END, "retry": "rewrite_query", "abstain": "abstain"},
    )
    g.add_edge("abstain", END)
    return g.compile()


app = build_app()


def run(question: str, **kwargs):
    """Convenience wrapper returning the final state for a single question."""
    final = {}
    for output in app.stream({"question": question, **kwargs}):
        final.update(next(iter(output.values())))
    return final


if __name__ == "__main__":
    for q in [
        "What is the main topic discussed in the provided document?",
        "What is the weather in Tokyo today?",
        "How many users signed up yesterday?",
    ]:
        print(f"\n=== {q} ===")
        result = run(q)
        print("Answer:", result.get("generation"))
        print("Confidence:", result.get("confidence"))
