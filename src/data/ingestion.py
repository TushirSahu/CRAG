"""Ingestion helpers for PDFs used by the CRAG pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.data.preprocessing import prepare_documents


def extract_documents_from_pdf(file_path: str | Path, source_name: str | None = None) -> list[Document]:
	"""Load a PDF with LlamaParse when available, otherwise fall back to PyPDFLoader."""

	path = Path(file_path)
	if not path.exists():
		raise FileNotFoundError(f"PDF not found: {path}")

	source = source_name or path.name
	llama_key = os.getenv("LLAMA_CLOUD_API_KEY")

	if llama_key:
		try:
			from llama_parse import LlamaParse

			parser = LlamaParse(api_key=llama_key, result_type="markdown", verbose=False)
			parsed_docs = parser.load_data(str(path))
			documents: list[Document] = []
			for index, parsed_doc in enumerate(parsed_docs):
				text = getattr(parsed_doc, "text", "") or str(parsed_doc)
				documents.append(
					Document(
						page_content=text,
						metadata={"source": source, "page": index + 1, "loader": "llamaparse"},
					)
				)
			return documents
		except Exception:
			pass

	from langchain_community.document_loaders import PyPDFLoader

	loader = PyPDFLoader(str(path))
	documents = loader.load()
	for doc in documents:
		doc.metadata = dict(doc.metadata or {})
		doc.metadata["source"] = source
		doc.metadata["loader"] = "pypdf"
	return documents


def ingest_pdf(file_path: str | Path, source_name: str | None = None, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
	"""Extract, clean, split, and return index-ready chunks."""

	extracted_documents = extract_documents_from_pdf(file_path=file_path, source_name=source_name)
	return prepare_documents(extracted_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
	"""Convert LangChain documents to plain dictionaries for storage or APIs."""

	return [{"content": doc.page_content, "metadata": dict(doc.metadata or {})} for doc in documents]
