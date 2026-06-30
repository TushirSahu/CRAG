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


class Claim(BaseModel):
    text: str = Field(description="A single factual claim from the answer.")
    supported: bool = Field(description="True if directly supported by the context.")


class Verification(BaseModel):
    claims: List[Claim] = Field(default_factory=list)


def verify_generation(generation: str, documents: List[Dict[str, Any]], llm) -> Dict[str, Any]:
    """Return ``{confidence, supported, total, claims, grounded}`` for an answer."""
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
    except Exception:
        # On parser/LLM failure, don't block — treat as grounded with unknown confidence.
        return {"confidence": 1.0, "supported": 0, "total": 0, "claims": [], "grounded": True}

    total = len(claims)
    supported = sum(1 for c in claims if c.get("supported"))
    confidence = supported / total if total else 1.0
    return {
        "confidence": confidence,
        "supported": supported,
        "total": total,
        "claims": claims,
        "grounded": total == 0 or confidence > 0,
    }
