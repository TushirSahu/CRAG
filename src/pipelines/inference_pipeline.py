"""Inference entrypoint for the CRAG LangGraph agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agent.graph import app as crag_app
from src.agent.semantic_cache import LocalSemanticCache


@dataclass
class InferenceResult:
	response: str
	documents: list[dict[str, Any]] = field(default_factory=list)
	route: list[str] = field(default_factory=list)
	cache_hit: bool = False


class CRAGInferencePipeline:
	"""Small wrapper that exposes the graph as a reusable inference API."""

	def __init__(self, use_cache: bool = True) -> None:
		self.use_cache = use_cache
		self.cache = LocalSemanticCache() if use_cache else None

	def invoke(self, question: str, chat_history: list[dict[str, str]] | None = None) -> InferenceResult:
		chat_history = chat_history or []

		if self.use_cache and self.cache is not None:
			cached_answer = self.cache.check_cache(question)
			if cached_answer:
				return InferenceResult(response=cached_answer, cache_hit=True)

		inputs = {"question": question, "retry_count": 0, "chat_history": chat_history}
		final_generation = ""
		final_documents: list[dict[str, Any]] = []
		route: list[str] = []

		for output in crag_app.stream(inputs):
			for key, value in output.items():
				route.append(key)
				if key == "generate":
					final_generation = value.get("generation", "")
					final_documents = value.get("documents", [])

		if self.use_cache and self.cache is not None and final_generation:
			self.cache.add_to_cache(question, final_generation)

		return InferenceResult(response=final_generation, documents=final_documents, route=route)

	def run(self, question: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
		"""Return a JSON-friendly response object."""

		result = self.invoke(question=question, chat_history=chat_history)
		return {
			"response": result.response,
			"documents": result.documents,
			"route": result.route,
			"cache_hit": result.cache_hit,
		}


def answer(question: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
	"""Convenience function for callers that do not want to manage a class instance."""

	return CRAGInferencePipeline().run(question=question, chat_history=chat_history)
