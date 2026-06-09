"""Phase D gate: the four LLM scenarios must all be green with the mock (zero $).

Behavior identical to old engine/tests/test_llm_scenarios.py.
Imports updated to use new app.services.* paths.

Run: pytest engine/src/tests -q
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.llm.verifier import LLMError, MockLLM, parse_results
from app.services.scoring import compute_hybrid_score, compute_verdict

SOURCE = {"product_id": 630, "name": "iPhone 15 Pro Max", "product_type": "Smartphone"}
CANDS = [
    {"product_id": 100 + i, "name": f"Cand {i}", "product_type": "Accessory"}
    for i in range(10)
]


def _run(coro):
    return asyncio.run(coro)


def test_scenario_valid_returns_parseable_results():
    """Scenario 1: valid -> well-formed results for every candidate, scorer works."""
    results, ti, to = _run(MockLLM("valid").verify(SOURCE, CANDS))
    assert len(results) == len(CANDS)
    for r in results:
        assert "candidate_id" in r and "logical_score" in r and "verdict" in r
        h = compute_hybrid_score(0.6, float(r["logical_score"]), 0.6)
        assert 0.0 <= h <= 1.0
        # verdict combines thresholds, never raises
        compute_verdict(0.6, float(r["logical_score"]), 0.3, 0.5)
    assert ti >= 0 and to >= 0


def test_scenario_slow_completes_after_delay():
    """Scenario 2: slow -> the call eventually returns; caller can wait/monitor."""
    llm = MockLLM("slow", slow_seconds=0.2)
    results, _, _ = _run(llm.verify(SOURCE, CANDS))
    assert len(results) == len(CANDS)


def test_scenario_slow_times_out_but_is_retryable():
    """Scenario 2b: with a tighter timeout the call times out; the job must NOT lose data -
    it raises TimeoutError which the worker treats as retryable (monitor until response)."""
    llm = MockLLM("slow", slow_seconds=1.0)
    with pytest.raises(asyncio.TimeoutError):
        _run(asyncio.wait_for(llm.verify(SOURCE, CANDS), timeout=0.1))


def test_scenario_error_raises_llmerror():
    """Scenario 3: error -> LLMError (worker retries, then dead-letters; no crash)."""
    with pytest.raises(LLMError):
        _run(MockLLM("error").verify(SOURCE, CANDS))


def test_scenario_truncated_is_handled_not_crash():
    """Scenario 4: truncated JSON -> parse yields [] (logged), no exception, no rows."""
    results, _, _ = _run(MockLLM("truncated").verify(SOURCE, CANDS))
    assert results == []


def test_parse_results_tolerates_garbage():
    assert parse_results("") == []
    assert parse_results("not json") == []
    assert parse_results('{"results": [{"candidate_id": 1}]}') == [{"candidate_id": 1}]
