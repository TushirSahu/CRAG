"""Validation helpers for uploaded PDFs and prepared CRAG documents."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_pdf_file(file_path: str | Path, min_pages: int = 1, min_size_bytes: int = 1) -> dict[str, Any]:
	"""Validate that a PDF exists and looks usable before ingestion."""

	path = Path(file_path)
	result: dict[str, Any] = {
		"path": str(path),
		"exists": path.exists(),
		"is_pdf": path.suffix.lower() == ".pdf",
		"size_bytes": path.stat().st_size if path.exists() else 0,
		"page_count": 0,
		"is_valid": False,
		"errors": [],
	}

	if not result["exists"]:
		result["errors"].append("File does not exist.")
		return result

	if not result["is_pdf"]:
		result["errors"].append("File must be a PDF.")

	if result["size_bytes"] < min_size_bytes:
		result["errors"].append("File is empty or too small.")

	try:
		from langchain_community.document_loaders import PyPDFLoader

		pages = PyPDFLoader(str(path)).load()
		result["page_count"] = len(pages)
		if result["page_count"] < min_pages:
			result["errors"].append(f"PDF has fewer than {min_pages} page(s).")
	except Exception as exc:  # pragma: no cover - depends on local PDF parser/runtime
		result["errors"].append(f"Unable to read PDF: {exc}")

	result["is_valid"] = len(result["errors"]) == 0
	return result


def validate_documents(documents: list[Any], min_chars: int = 30) -> dict[str, Any]:
	"""Check whether extracted documents contain enough usable text."""

	total_chars = 0
	usable_docs = 0
	errors: list[str] = []

	for doc in documents:
		content = getattr(doc, "page_content", None)
		if content is None and isinstance(doc, dict):
			content = doc.get("content", "")
		text = str(content or "").strip()
		total_chars += len(text)
		if len(text) >= min_chars:
			usable_docs += 1

	if not documents:
		errors.append("No documents were extracted.")
	if usable_docs == 0:
		errors.append("Extracted documents are too short to be useful.")

	return {
		"document_count": len(documents),
		"usable_documents": usable_docs,
		"total_chars": total_chars,
		"is_valid": len(errors) == 0,
		"errors": errors,
	}
