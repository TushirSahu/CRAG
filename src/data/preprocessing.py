"""Text cleaning and chunk-preparation utilities for CRAG documents."""

from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def normalize_text(text: str) -> str:
	"""Normalize whitespace and remove noisy control characters."""

	text = text.replace("\u00ad", "")
	text = re.sub(r"\s+", " ", text)
	return text.strip()


def clean_documents(documents: Iterable[Document]) -> list[Document]:
	"""Return documents with normalized text and preserved metadata."""

	cleaned_docs: list[Document] = []
	for doc in documents:
		content = normalize_text(getattr(doc, "page_content", "") or "")
		metadata = dict(getattr(doc, "metadata", {}) or {})
		if content:
			cleaned_docs.append(Document(page_content=content, metadata=metadata))
	return cleaned_docs


def split_documents(documents: Iterable[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
	"""Split cleaned documents into retrieval-friendly chunks."""

	splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	return splitter.split_documents(list(documents))


def deduplicate_documents(documents: Iterable[Document]) -> list[Document]:
	"""Remove duplicate chunks while preserving the first occurrence."""

	unique_docs: list[Document] = []
	seen: set[tuple[str, tuple[tuple[str, str], ...]]] = set()

	for doc in documents:
		metadata_items = tuple(sorted((str(key), str(value)) for key, value in (doc.metadata or {}).items()))
		fingerprint = (normalize_text(doc.page_content).lower(), metadata_items)
		if fingerprint not in seen:
			seen.add(fingerprint)
			unique_docs.append(doc)
	return unique_docs


def prepare_documents(documents: Iterable[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
	"""Clean, split, and deduplicate documents for indexing."""

	cleaned_docs = clean_documents(documents)
	chunked_docs = split_documents(cleaned_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	return deduplicate_documents(chunked_docs)
