"""Tests for the fail-*safe* trust layer and the post-verify routing.

A fake LLM (a ``RunnableLambda`` returning a canned string) drives
``verify_generation`` without any API call.
"""

from langchain_core.runnables import RunnableLambda

from src.agent import nodes
from src.agent.trust import verify_generation

DOCS = [{"content": "The sky is blue."}]


def _llm(text):
    return RunnableLambda(lambda _x: text)


def test_unparseable_response_is_unverified_not_perfect():
    # The old code returned confidence 1.0 here — the fail-open bug.
    report = verify_generation("The sky is blue.", DOCS, _llm("not json at all"))
    assert report["verified"] is False
    assert report["confidence"] == 0.0
    assert report["grounded"] is False


def test_zero_claims_is_not_grounded():
    report = verify_generation("", DOCS, _llm('{"claims": []}'))
    assert report["verified"] is True
    assert report["confidence"] == 0.0
    assert report["grounded"] is False


def test_partial_support_yields_fractional_confidence():
    payload = '{"claims": [{"text": "a", "supported": true}, {"text": "b", "supported": false}]}'
    report = verify_generation("a and b", DOCS, _llm(payload))
    assert report["confidence"] == 0.5
    assert report["verified"] is True
    assert report["grounded"] is True


def test_decide_after_verify_routes():
    assert nodes.decide_after_verify({"verified": True, "confidence": 0.9}) == "useful"
    assert nodes.decide_after_verify({"verified": True, "confidence": 0.1, "retry_count": 0}) == "retry"
    # retries exhausted: verified-but-weak abstains, unverified is surfaced.
    assert nodes.decide_after_verify({"verified": True, "confidence": 0.1, "retry_count": 9}) == "abstain"
    assert nodes.decide_after_verify({"verified": False, "confidence": 0.0, "retry_count": 9}) == "unverified"


if __name__ == "__main__":
    test_unparseable_response_is_unverified_not_perfect()
    test_zero_claims_is_not_grounded()
    test_partial_support_yields_fractional_confidence()
    test_decide_after_verify_routes()
    print("✅ trust tests passed")
