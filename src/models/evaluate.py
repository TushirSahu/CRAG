"""Evaluation utilities for the CRAG system.

This module keeps evaluation lightweight and dependency-safe:
- it can run against the live LangGraph agent,
- it produces local proxy metrics when Ragas is unavailable,
- and it can persist a JSON report for CI or manual review.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from src.agent.graph import app as crag_app


@dataclass
class EvaluationExample:
	question: str
	reference: str | None = None
	chat_history: list[dict[str, str]] | None = None


def _tokenize(text: str) -> set[str]:
	tokens = re.findall(r"[a-z0-9]+", text.lower())
	stopwords = {
		"the",
		"and",
		"for",
		"with",
		"that",
		"this",
		"what",
		"when",
		"where",
		"which",
		"from",
		"into",
		"about",
		"does",
		"are",
		"was",
		"were",
		"has",
		"have",
		"had",
		"who",
		"why",
		"how",
		"what",
	}
	return {token for token in tokens if len(token) > 2 and token not in stopwords}


def _extract_generation_and_documents(question: str, chat_history: list[dict[str, str]] | None = None) -> tuple[str, list[dict[str, Any]], list[str]]:
	inputs = {"question": question, "retry_count": 0, "chat_history": chat_history or []}
	generation = ""
	documents: list[dict[str, Any]] = []
	route: list[str] = []

	for output in crag_app.stream(inputs):
		for key, value in output.items():
			route.append(key)
			if key == "generate":
				generation = value.get("generation", "")
				documents = value.get("documents", [])

	return generation, documents, route


def _score_overlap(left: str, right: str) -> float:
	left_tokens = _tokenize(left)
	right_tokens = _tokenize(right)
	if not left_tokens or not right_tokens:
		return 0.0
	return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def score_example(example: EvaluationExample) -> dict[str, Any]:
	"""Run the live agent and compute lightweight quality signals."""

	answer, documents, route = _extract_generation_and_documents(example.question, example.chat_history)
	context_text = " ".join(doc.get("content", "") for doc in documents)

	question_tokens = _tokenize(example.question)
	answer_tokens = _tokenize(answer)
	context_tokens = _tokenize(context_text)
	reference_tokens = _tokenize(example.reference or "")

	answer_relevancy = _score_overlap(example.question, answer)
	context_precision = len(question_tokens & context_tokens) / max(len(context_tokens), 1)
	context_recall = len(reference_tokens & context_tokens) / max(len(reference_tokens), 1) if reference_tokens else 0.0
	faithfulness = 1.0 if answer and any(token in context_tokens for token in answer_tokens) else 0.0

	return {
		"question": example.question,
		"reference": example.reference,
		"answer": answer,
		"retrieved_documents": documents,
		"route": route,
		"metrics": {
			"answer_relevancy": round(answer_relevancy, 4),
			"context_precision": round(context_precision, 4),
			"context_recall": round(context_recall, 4),
			"faithfulness": round(faithfulness, 4),
		},
	}


def evaluate_examples(examples: Iterable[EvaluationExample]) -> dict[str, Any]:
	"""Evaluate multiple examples and aggregate average metrics."""

	results = [score_example(example) for example in examples]
	metric_keys = ["answer_relevancy", "context_precision", "context_recall", "faithfulness"]
	totals = {key: 0.0 for key in metric_keys}

	for result in results:
		for key in metric_keys:
			totals[key] += float(result["metrics"][key])

	count = max(len(results), 1)
	summary = {key: round(total / count, 4) for key, total in totals.items()}

	return {"count": len(results), "summary": summary, "results": results}


def save_report(report: dict[str, Any], output_path: str | Path) -> str:
	"""Persist an evaluation report as JSON."""

	path = Path(output_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(report, indent=2), encoding="utf-8")
	return str(path)


def _load_examples_from_json(path: str | Path) -> list[EvaluationExample]:
	raw_items = json.loads(Path(path).read_text(encoding="utf-8"))
	examples: list[EvaluationExample] = []
	for item in raw_items:
		examples.append(
			EvaluationExample(
				question=item["question"],
				reference=item.get("reference"),
				chat_history=item.get("chat_history"),
			)
		)
	return examples


def build_default_examples() -> list[EvaluationExample]:
	"""Small smoke-test set that matches the CRAG use cases."""

	return [
		EvaluationExample(
			question="What does the report explicitly state regarding ASC 842 and the classification of operating lease liabilities?",
			reference="Operating lease liabilities are recognized on the balance sheet under ASC 842 based on the present value of future lease payments.",
		),
		EvaluationExample(
			question="According to the signatures on the Form 10-K, who serves as the Chief Financial Officer and Principal Accounting Officer?",
			reference="Vaibhav Taneja serves as the Chief Financial Officer and Principal Accounting Officer.",
		),
	]


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate the CRAG agent on a small test set.")
	parser.add_argument("--input", type=str, help="Path to a JSON file with evaluation examples.")
	parser.add_argument("--output", type=str, default="./reports/rag_evaluation_report.json", help="Where to save the JSON report.")
	args = parser.parse_args()

	examples = _load_examples_from_json(args.input) if args.input else build_default_examples()
	report = evaluate_examples(examples)
	saved_path = save_report(report, args.output)
	print(f"Saved evaluation report to {saved_path}")
	print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
	main()
