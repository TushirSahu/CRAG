"""Verifiable trust layer (Pillar B).

Turns the old single yes/no hallucination check into an auditable signal:
break the answer into claims, check each against the retrieved context in one
structured LLM call, and return a confidence score. The graph uses this to
**abstain** instead of emitting an ungrounded answer.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from src.agent.prompts import VERIFY_GENERATION
from src.utils.logger import logger


class Claim(BaseModel):
    text: str = Field(description="A single factual claim from the answer.")
    supported: bool = Field(description="True if directly supported by the context.")


class Verification(BaseModel):
    claims: List[Claim] = Field(default_factory=list)


def _report(confidence: float, supported: int, total: int, claims: list, verified: bool) -> Dict[str, Any]:
    return {
        "confidence": confidence,
        "supported": supported,
        "total": total,
        "claims": claims,
        "verified": verified,                 # did the verification check itself run?
        "grounded": verified and confidence > 0,
    }


def verify_generation(generation: str, documents: List[Dict[str, Any]], llm) -> Dict[str, Any]:
    """Score how well ``generation`` is grounded in ``documents``.

    Returns ``{confidence, supported, total, claims, verified, grounded}``. Fails
    **safe**: if the check cannot run (LLM/parser error) or the answer has no
    verifiable claims, confidence is ``0.0`` and ``verified`` flags that the
    answer is *unverified* — never silently treated as fully grounded.
    """
    parser = JsonOutputParser(pydantic_object=Verification)
    docs_str = "\n\n".join(d["content"] for d in documents) or "(no context)"

    try:
        result = (VERIFY_GENERATION | llm | parser).invoke(
            {
                "documents": docs_str,
                "generation": generation,
                "format_instructions": parser.get_format_instructions(),
            }
        )
        claims = result.get("claims", []) if isinstance(result, dict) else []
    except Exception as exc:
        # Can't verify → don't claim grounding. Surface it as unverified, not perfect.
        logger.warning("Trust check failed, treating answer as unverified: %s", exc)
        return _report(confidence=0.0, supported=0, total=0, claims=[], verified=False)

    total = len(claims)
    supported = sum(1 for c in claims if c.get("supported"))
    # No extractable claims means nothing to ground on → not grounded.
    confidence = supported / total if total else 0.0
    return _report(confidence, supported, total, claims, verified=True)
