"""Tests for MockJudge, judge_pair, select_sample, and the ground_truth -> run_eval path.

All tests are $0/mock: no real OpenAI calls, no real DB connections.
DB interactions use StubRepo (in-memory).
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.compatibility.judge import MockJudge, judge_pair, select_sample
from app.services.eval.evaluator import build_snapshot_row


# ---------------------------------------------------------------------------
# Stub repository
# ---------------------------------------------------------------------------


class StubJudgeRepo:
    """Minimal MetricsRepository stub for judge tests."""

    def __init__(self, pairs: list[dict] | None = None) -> None:
        # pre-loaded pairs returned by sample_pairs_for_judge
        self._pairs = pairs or []
        self.gt_rows: list[dict] = []

    async def sample_pairs_for_judge(
        self, experiment_id: str, n: int = 500, stratify_by: str = "category"
    ) -> list[dict]:
        return self._pairs[:n]

    async def write_ground_truth(self, rows: list[dict]) -> int:
        self.gt_rows.extend(rows)
        return len(rows)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# MockJudge unit tests
# ---------------------------------------------------------------------------


def test_mock_judge_returns_valid_label():
    """MockJudge.judge_pair() always returns label 0 or 1."""
    judge = MockJudge()
    src = {"product_id": 1, "name": "Coffee Machine"}
    cand = {"product_id": 2, "name": "Descaler"}
    row = _run(judge.judge_pair(src, cand))
    assert row["label"] in (0, 1)


def test_mock_judge_is_deterministic():
    """Same pair -> same label on repeated calls."""
    judge = MockJudge()
    src = {"product_id": 10}
    cand = {"product_id": 20}
    row1 = _run(judge.judge_pair(src, cand))
    row2 = _run(judge.judge_pair(src, cand))
    assert row1["label"] == row2["label"]


def test_mock_judge_varies_labels():
    """MockJudge produces a mix of 0 and 1 labels across pairs (not all same)."""
    judge = MockJudge(positive_rate=0.5)
    labels = set()
    for i in range(20):
        src = {"product_id": i}
        cand = {"product_id": i + 100}
        row = _run(judge.judge_pair(src, cand))
        labels.add(row["label"])
    # With 20 diverse pairs and positive_rate=0.5 we expect both labels
    assert len(labels) == 2, f"Expected both 0 and 1, got labels={labels}"


def test_mock_judge_row_shape():
    """MockJudge row has all required keys."""
    judge = MockJudge(judge_model="test_model")
    src = {"product_id": 5, "name": "Machine"}
    cand = {"product_id": 6, "name": "Filter"}
    row = _run(judge.judge_pair(src, cand))

    required_keys = {
        "product_i",
        "product_j",
        "label",
        "source",
        "judge_model",
        "rationale",
    }
    assert required_keys.issubset(row.keys()), (
        f"Missing keys: {required_keys - row.keys()}"
    )
    assert row["product_i"] == 5
    assert row["product_j"] == 6
    assert row["judge_model"] == "test_model"
    assert row["source"] == "llm"
    assert isinstance(row["rationale"], str)
    assert len(row["rationale"]) > 0


# ---------------------------------------------------------------------------
# judge_pair with MockLLM
# ---------------------------------------------------------------------------


class MockLLMForJudge:
    """Returns deterministic {'label': 1, 'rationale': 'test'} for all pairs."""

    async def generate(self, system_prompt, user_prompt, response_format):
        return {"label": 1, "rationale": "Mock: technically compatible."}, 10, 10


def test_judge_pair_parses_llm_response():
    """judge_pair() correctly parses dict response from mock LLM."""
    src = {"product_id": 1, "name": "Coffee Machine", "product_type": "coffee_machine"}
    cand = {"product_id": 2, "name": "Descaler", "product_type": "descaler"}
    llm = MockLLMForJudge()
    row = _run(judge_pair(src, cand, llm, judge_model="mock_v1"))

    assert row["label"] == 1
    assert row["rationale"] == "Mock: technically compatible."
    assert row["product_i"] == 1
    assert row["product_j"] == 2
    assert row["judge_model"] == "mock_v1"


class MockLLMBadResponse:
    """Returns malformed response (missing label) to test error handling."""

    async def generate(self, system_prompt, user_prompt, response_format):
        return {"rationale": "no label key"}, 10, 10


def test_judge_pair_raises_on_missing_label():
    """judge_pair() raises ValueError when label is absent or invalid."""
    src = {"product_id": 1}
    cand = {"product_id": 2}
    llm = MockLLMBadResponse()
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        _run(judge_pair(src, cand, llm))


# ---------------------------------------------------------------------------
# select_sample delegates to repo
# ---------------------------------------------------------------------------


def test_select_sample_returns_repo_pairs():
    """select_sample() forwards the repo result up to n pairs."""
    pairs = [
        {
            "product_i": i,
            "product_j": i + 100,
            "context_code": "accessory",
            "semantic_score": 0.7,
            "logical_score": 0.6,
            "hybrid_score": 0.65,
        }
        for i in range(20)
    ]
    repo = StubJudgeRepo(pairs=pairs)
    result = _run(select_sample(repo, "baseline_v1", n=10))
    assert len(result) == 10


def test_select_sample_respects_n():
    """select_sample() respects the n limit even if repo has more pairs."""
    pairs = [
        {
            "product_i": i,
            "product_j": i + 100,
            "context_code": None,
            "semantic_score": 0.5,
            "logical_score": 0.5,
            "hybrid_score": 0.5,
        }
        for i in range(100)
    ]
    repo = StubJudgeRepo(pairs=pairs)
    result = _run(select_sample(repo, "exp1", n=15))
    assert len(result) == 15


# ---------------------------------------------------------------------------
# End-to-end: mock judge -> ground_truth -> build_snapshot_row has non-null P/R
# ---------------------------------------------------------------------------


def test_mock_judge_to_ground_truth_enables_precision_recall():
    """Mock judge produces ground_truth rows that make P@10 non-null in evaluator.

    This test proves the full pipeline:
      MockJudge -> gt_rows -> build_snapshot_row with matching recommendations
      -> precision_at_10 is not None.
    """
    # 10 recommendation rows for product_id=1 (ids 100..109), sorted by hybrid DESC.
    recs = [
        {
            "product_id": 1,
            "recommended_id": 100 + i,
            "semantic_score": 0.8 - i * 0.02,
            "logical_score": 0.7 - i * 0.02,
            "hybrid_score": 0.75 - i * 0.02,
            "verdict": 1 if i < 5 else 0,
        }
        for i in range(10)
    ]

    # Build ground_truth via MockJudge for the same pairs.
    judge = MockJudge(positive_rate=0.5)
    gt_map: dict[int, set[int]] = {1: set()}
    for i in range(10):
        src = {"product_id": 1}
        cand = {"product_id": 100 + i}
        row = _run(judge.judge_pair(src, cand))
        if row["label"] == 1:
            gt_map[1].add(100 + i)

    # Ensure at least one positive label for the test to be meaningful.
    if not gt_map[1]:
        gt_map[1].add(100)  # force at least one positive

    snapshot, _ = build_snapshot_row(
        experiment_id="test_judge_exp",
        pairs=recs,
        gt_map=gt_map,
        tau_s=0.3,
        tau_l=0.5,
    )

    assert snapshot["precision_at_10"] is not None, (
        "precision@10 must be non-null when ground_truth rows exist"
    )
    assert snapshot["recall_at_10"] is not None, (
        "recall@10 must be non-null when ground_truth rows exist"
    )
    assert 0.0 <= snapshot["precision_at_10"] <= 1.0
    assert 0.0 <= snapshot["recall_at_10"] <= 1.0
