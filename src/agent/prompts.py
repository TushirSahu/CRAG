"""All LLM prompt templates, lifted out of node function bodies."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ROUTER = ChatPromptTemplate.from_template(
    "You are an expert intent router. Classify the user query into exactly one bucket:\n"
    "1. 'vector': policies, documents, or general knowledge in our knowledge base.\n"
    "2. 'web': live information, current weather, or recent news.\n"
    "3. 'sql': structured data, counts, or analytics.\n"
    "Return strictly ONE word: 'vector', 'web', or 'sql'.\n"
    "Question: {question}"
)

GRADE_DOCUMENTS = ChatPromptTemplate.from_template(
    "You are grading the relevance of a retrieved document to a user question.\n"
    "Question: {question}\n"
    "Document: {document}\n"
    "If the document shares keywords or semantic meaning with the question, it is relevant.\n"
    "Respond with a single word: 'yes' or 'no'."
)

REWRITE_QUERY = ChatPromptTemplate.from_template(
    "You are a search-query optimizer. Rewrite the user's question into a single, "
    "highly effective search-engine query with no conversational text.\n"
    "{format_instructions}\n"
    "Initial question: {question}"
)

SUMMARIZE_CLUSTER = ChatPromptTemplate.from_template(
    "Write a concise, information-dense summary of the following related passages. "
    "Preserve key facts, entities, and numbers so the summary can answer high-level "
    "questions on its own.\n\nPassages:\n{passages}\n\nSummary:"
)

GENERATE = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. Be detailed and comprehensive, "
    "and synthesize across sources. Cite inline as [Source 1], [Source 2], etc. "
    "If a source is marked (stale), prefer fresher sources and note when information may "
    "be outdated. End with a '### References' section listing each cited source EXACTLY "
    "as written in the context blocks. Do not invent hyperlinks or add apologetic filler.\n\n"
    "{history_text}\n"
    "Context:\n{context}\n\n"
    "Question: {question}"
)

TEXT_TO_SQL = ChatPromptTemplate.from_template(
    "You are a careful analytics engineer. Given a SQLite schema and a question, "
    "write ONE read-only SQL query that answers it.\n"
    "Rules: a single SELECT statement only; no INSERT/UPDATE/DELETE/DDL; no semicolons "
    "beyond the one terminating the statement; use only the tables and columns shown.\n"
    "Today's date is {today} (use it for relative dates like 'yesterday').\n\n"
    "Schema:\n{schema}\n\n"
    "Question: {question}\n"
    "{format_instructions}"
)

VERIFY_GENERATION = ChatPromptTemplate.from_template(
    "You are a strict fact-checker. For the answer below, break it into its factual "
    "claims and decide, for each, whether it is directly supported by the provided "
    "context. Ignore citations, headers, and filler.\n"
    "Context:\n-----\n{documents}\n-----\n"
    "Answer:\n-----\n{generation}\n-----\n"
    "{format_instructions}"
)
