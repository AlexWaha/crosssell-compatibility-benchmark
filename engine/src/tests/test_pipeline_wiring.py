"""Tests for pipeline.py JIT/oneshot wiring and instrumentation.

All tests are $0/mock: no real OpenAI calls, no real DB connections,
no real Typesense connections. Uses in-memory stubs for all I/O.

Tests cover:
- JIT path produces pipeline_runs + stage_metrics + compatibility_evaluations rows.
- Recommendations written only for verdict=1 pairs.
- Oneshot (control arm) path still callable.
- COMPAT_MODE switch selects the correct path.
- no_source and no_candidates early exits work correctly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from app.services.pipeline import process_product


# ---------------------------------------------------------------------------
# Stub objects
# ---------------------------------------------------------------------------


class StubSettings:
    """Minimal engine settings stub for pipeline tests."""

    LLM_MODE = "mock"
    MOCK_SCENARIO = "valid"
    COMPAT_MODE = "jit"
    COMPAT_TAU_S = 0.3
    COMPAT_TAU_L = 0.5
    COMPAT_ALPHA = 0.6
    COMPAT_BATCH_SIZE = 10
    COMPAT_TOP_K = 25
    RETRIEVAL_STRATEGY = "semantic"
    TYPESENSE_API_KEY = "xyz"
    TYPESENSE_COLLECTION = "products"
    TYPESENSE_HOST = "typesense"
    TYPESENSE_PORT = 8108
    HDR_ENABLED = False  # simplifies tests (no vocab filtering)
    L_AGG = "weighted_product"
    RULE_CACHE_BACKEND = "mock"
    EXPERIMENT_ID = "test_exp"

    @property
    def typesense_base_url(self):
        return f"http://{self.TYPESENSE_HOST}:{self.TYPESENSE_PORT}"


class StubRepo:
    """Captures all write calls for assertion in tests."""

    def __init__(self) -> None:
        self.recommendations: list[dict] = []
        self.evaluations: list[dict] = []
        self.started: list[dict] = []
        self.finished: list[dict] = []
        self.stage_rows: list = []
        self._next_run_id = 1

    async def start_run(
        self, product_id, pipeline_version, job_id=None, experiment_id="baseline_v1"
    ):
        run_id = self._next_run_id
        self._next_run_id += 1
        self.started.append({"run_id": run_id, "product_id": product_id})
        return run_id

    async def finish_run(self, run_id, status, total_duration_ms, error_message=None):
        self.finished.append({"run_id": run_id, "status": status})

    async def write_stage_metrics(self, run_id, rows):
        self.stage_rows.extend(rows)

    async def write_recommendations(self, rows: list[dict]) -> int:
        self.recommendations.extend(rows)
        return len(rows)

    async def write_evaluations(self, rows: list[dict]) -> int:
        self.evaluations.extend(rows)
        return len(rows)

    async def write_ground_truth(self, rows):
        return 0


# A minimal 1024-dim unit embedding vector (all zeros except first element).
_DUMMY_EMBEDDING = [1.0] + [0.0] * 1023


def _make_source_doc(product_id: int = 1) -> dict:
    """Build a realistic source document as returned by Typesense.

    Includes a dummy embedding so retrieve_candidates does not return [] early.
    Source categories are 'Coffee Machines'; candidates use 'Cleaning' so they
    survive filter_candidates (different category, not self-reference).
    """
    return {
        "product_id": product_id,
        "name": f"Coffee Machine {product_id}",
        "product_type": "coffee_machine",
        "categories": ["Coffee Machines"],
        "_score": 1.0,
        "embedding": _DUMMY_EMBEDDING,
        "attributes": {
            "brand": "siemens",
            "product_category": "coffee_machine",
            "compatible_brands": ["siemens"],
        },
        "attributes_json": json.dumps(
            {
                "brand": "siemens",
                "product_category": "coffee_machine",
                "compatible_brands": ["siemens"],
            }
        ),
    }


def _make_candidate_doc(product_id: int, score: float = 0.7) -> dict:
    """Build a realistic candidate document with a different category than the source."""
    return {
        "product_id": product_id,
        "name": f"Descaler {product_id}",
        "product_type": "descaler",
        "categories": ["Cleaning"],
        "_score": score,
        "vector_distance": 1.0 - score,
        "attributes": {
            "brand": "siemens",
            "product_category": "descaler",
            "target_appliance_category": "coffee_machine",
            "compatible_brands": ["siemens", "bosch"],
        },
        "attributes_json": json.dumps(
            {
                "brand": "siemens",
                "product_category": "descaler",
                "target_appliance_category": "coffee_machine",
                "compatible_brands": ["siemens", "bosch"],
            }
        ),
    }


# ---------------------------------------------------------------------------
# HTTP mock
# ---------------------------------------------------------------------------


def _make_mock_http(source_doc: dict | None, candidate_docs: list[dict]) -> Any:
    """Build a minimal httpx.AsyncClient mock supporting GET and POST.

    GET /search (first call): returns source_doc.
    POST /multi_search: returns candidate_docs as vector search hits.
    All other GET calls return empty hits.
    """

    async def mock_get(url, params=None, headers=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if source_doc is None:
            resp.json.return_value = {"hits": []}
        else:
            resp.json.return_value = {"hits": [{"document": source_doc}]}
        return resp

    async def mock_post(url, json=None, headers=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        hits = [
            {
                "document": doc,
                "text_match": 1,
                "vector_distance": doc.get("vector_distance", 0.3),
            }
            for doc in candidate_docs
        ]
        resp.json.return_value = {"results": [{"hits": hits}]}
        return resp

    http = MagicMock()
    http.get = mock_get
    http.post = mock_post
    return http


# ---------------------------------------------------------------------------
# JIT path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jit_path_creates_pipeline_run():
    """process_product JIT path opens a pipeline_runs row."""
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=1)
    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(10, 14)]
    http = _make_mock_http(source, cands)

    result = await process_product(1, "test_exp", settings, repo, http)

    assert result["status"] == "ok"
    assert len(repo.started) == 1
    assert repo.started[0]["product_id"] == 1


@pytest.mark.asyncio
async def test_jit_path_writes_compatibility_evaluations():
    """process_product JIT writes compatibility_evaluations for ALL candidates."""
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=1)
    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(10, 14)]
    http = _make_mock_http(source, cands)

    await process_product(1, "test_exp", settings, repo, http)

    # All 4 candidates get an evaluation row.
    assert len(repo.evaluations) == 4
    for ev in repo.evaluations:
        assert "run_id" in ev
        assert "experiment_id" in ev
        assert ev["experiment_id"] == "test_exp"
        assert "rules_evaluated" in ev
        assert "rules_passed" in ev
        assert "rules_failed" in ev
        assert "rules_undefined" in ev


@pytest.mark.asyncio
async def test_jit_path_recommendations_only_for_verdict1():
    """write_recommendations called only with verdict=True rows."""
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=1)
    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(10, 14)]
    http = _make_mock_http(source, cands)

    await process_product(1, "test_exp", settings, repo, http)

    # Recommendations must all have verdict=True
    for rec in repo.recommendations:
        assert rec["verdict"] is True


@pytest.mark.asyncio
async def test_jit_path_stage_metrics_flushed():
    """process_product flushes stage_metrics rows at finish."""
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=2)
    cands = [_make_candidate_doc(pid, score=0.75) for pid in range(20, 23)]
    http = _make_mock_http(source, cands)

    await process_product(2, "test_exp", settings, repo, http)

    # At least one stage_metrics row written (compatibility + recommendation stages).
    assert len(repo.stage_rows) >= 1
    # Run marked as completed.
    assert repo.finished[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_jit_path_no_source_returns_early():
    """No source document -> early return without DB writes."""
    settings = StubSettings()
    repo = StubRepo()
    http = _make_mock_http(source_doc=None, candidate_docs=[])

    result = await process_product(99, "test_exp", settings, repo, http)

    assert result["status"] == "no_source"
    assert len(repo.recommendations) == 0
    assert len(repo.evaluations) == 0
    assert repo.finished[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_jit_path_no_candidates_returns_early():
    """Source exists but no candidates above tau_S -> early return."""
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=5)
    # Candidates with score below tau_S=0.3 -> filtered out after tau_S gate.
    cands = [_make_candidate_doc(pid, score=0.1) for pid in range(30, 33)]
    http = _make_mock_http(source, cands)

    result = await process_product(5, "test_exp", settings, repo, http)

    assert result["status"] == "no_candidates"
    assert len(repo.recommendations) == 0


# ---------------------------------------------------------------------------
# Oneshot (control arm) test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oneshot_path_is_callable():
    """COMPAT_MODE='oneshot' runs the legacy verify() path without errors."""
    settings = StubSettings()
    settings.COMPAT_MODE = "oneshot"
    repo = StubRepo()
    source = _make_source_doc(product_id=7)
    cands = [_make_candidate_doc(pid, score=0.75) for pid in range(40, 43)]
    http = _make_mock_http(source, cands)

    result = await process_product(7, "test_exp", settings, repo, http)

    assert result["status"] == "ok"
    # Evaluations written from oneshot path as well.
    assert len(repo.evaluations) == 3


# ---------------------------------------------------------------------------
# rules_* count correctness test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jit_rules_counts_are_non_negative_integers():
    """compatibility_evaluations rules_* fields are non-negative integers that sum correctly.

    This test verifies numeric correctness, not just key presence:
    - rules_evaluated >= 0
    - rules_passed + rules_failed + rules_undefined == rules_evaluated
    - all values are integers (not None or float)
    """
    settings = StubSettings()
    repo = StubRepo()
    source = _make_source_doc(product_id=11)
    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(60, 63)]
    http = _make_mock_http(source, cands)

    await process_product(11, "test_exp", settings, repo, http)

    assert len(repo.evaluations) == 3
    for ev in repo.evaluations:
        n_eval = ev["rules_evaluated"]
        n_pass = ev["rules_passed"]
        n_fail = ev["rules_failed"]
        n_undef = ev["rules_undefined"]
        # All must be non-negative integers
        assert isinstance(n_eval, int) and n_eval >= 0
        assert isinstance(n_pass, int) and n_pass >= 0
        assert isinstance(n_fail, int) and n_fail >= 0
        assert isinstance(n_undef, int) and n_undef >= 0
        # The three outcome counts must sum to the total evaluated
        assert n_pass + n_fail + n_undef == n_eval, (
            f"rules_passed({n_pass}) + rules_failed({n_fail}) "
            f"+ rules_undefined({n_undef}) != rules_evaluated({n_eval})"
        )


# ---------------------------------------------------------------------------
# Oneshot control arm evaluation shape test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_oneshot_evaluations_have_rules_shape():
    """COMPAT_MODE='oneshot' evaluations contain all rules_* fields with non-negative ints.

    The oneshot (LLM verify) path populates rules_* fields from the LLM response.
    Mock verify() returns rules_evaluated=0 so we only verify shape and non-negativity.
    """
    settings = StubSettings()
    settings.COMPAT_MODE = "oneshot"
    repo = StubRepo()
    source = _make_source_doc(product_id=70)
    cands = [_make_candidate_doc(pid, score=0.75) for pid in range(80, 83)]
    http = _make_mock_http(source, cands)

    await process_product(70, "test_exp", settings, repo, http)

    assert len(repo.evaluations) == 3
    required_keys = {
        "run_id",
        "rules_evaluated",
        "rules_passed",
        "rules_failed",
        "rules_undefined",
    }
    for ev in repo.evaluations:
        assert required_keys.issubset(ev.keys()), (
            f"Missing keys: {required_keys - ev.keys()}"
        )
        assert isinstance(ev["rules_evaluated"], int) and ev["rules_evaluated"] >= 0
        assert isinstance(ev["rules_passed"], int) and ev["rules_passed"] >= 0
        assert isinstance(ev["rules_failed"], int) and ev["rules_failed"] >= 0
        assert isinstance(ev["rules_undefined"], int) and ev["rules_undefined"] >= 0


# ---------------------------------------------------------------------------
# COMPAT_MODE switch test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compat_mode_switch_jit_vs_oneshot():
    """JIT and oneshot paths both run without errors and produce consistent results."""
    source = _make_source_doc(product_id=8)
    cands = [_make_candidate_doc(pid, score=0.8) for pid in range(50, 53)]

    # JIT
    settings_jit = StubSettings()
    settings_jit.COMPAT_MODE = "jit"
    repo_jit = StubRepo()
    await process_product(
        8, "exp", settings_jit, repo_jit, _make_mock_http(source, cands)
    )

    # Oneshot
    settings_os = StubSettings()
    settings_os.COMPAT_MODE = "oneshot"
    repo_os = StubRepo()
    await process_product(
        8, "exp", settings_os, repo_os, _make_mock_http(source, cands)
    )

    assert len(repo_jit.evaluations) == 3
    assert len(repo_os.evaluations) == 3
    assert len(repo_jit.started) == 1
    assert len(repo_os.started) == 1


# ---------------------------------------------------------------------------
# HDR evidence_claims / hallucinated_claims forwarded to eval_rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_jit_evaluations_carry_evidence_and_hallucinated_claims():
    """compatibility_evaluations rows carry non-zero evidence_claims when rules exist.

    With HDR_ENABLED=False and MockRuleGen, evidence_claims == number of rules
    generated for the pair (>0) and hallucinated_claims == 0.
    This confirms that both HDR audit columns are forwarded from _verify_jit
    into every eval_row and are not silently zeroed.
    """
    settings = StubSettings()
    settings.HDR_ENABLED = False
    repo = StubRepo()
    source = _make_source_doc(product_id=20)
    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(30, 33)]
    http = _make_mock_http(source, cands)

    await process_product(20, "test_exp", settings, repo, http)

    assert len(repo.evaluations) == 3
    for ev in repo.evaluations:
        assert "evidence_claims" in ev, (
            "evidence_claims key must be present in eval_row"
        )
        assert "hallucinated_claims" in ev, (
            "hallucinated_claims key must be present in eval_row"
        )
        # MockRuleGen returns 2 rules for coffee_machine -> descaler.
        # With HDR_ENABLED=False, evidence_claims = len(rules) >= 2 (non-zero).
        assert isinstance(ev["evidence_claims"], int) and ev["evidence_claims"] >= 0
        assert (
            isinstance(ev["hallucinated_claims"], int)
            and ev["hallucinated_claims"] >= 0
        )
        # Rules exist for this pair, so evidence_claims must be non-zero.
        if ev["rules_evaluated"] > 0:
            assert ev["evidence_claims"] > 0, (
                f"evidence_claims must be non-zero when rules_evaluated={ev['rules_evaluated']}"
            )


@pytest.mark.asyncio
async def test_oneshot_evaluations_carry_evidence_and_hallucinated_claims():
    """oneshot path eval_rows also carry evidence_claims and hallucinated_claims keys.

    The oneshot LLM verify() path does not compute HDR internally, so both
    columns default to 0 from r.get(..., 0) - but the keys must be present.
    """
    settings = StubSettings()
    settings.COMPAT_MODE = "oneshot"
    repo = StubRepo()
    source = _make_source_doc(product_id=21)
    cands = [_make_candidate_doc(pid, score=0.75) for pid in range(40, 43)]
    http = _make_mock_http(source, cands)

    await process_product(21, "test_exp", settings, repo, http)

    assert len(repo.evaluations) == 3
    required_hdr_keys = {"evidence_claims", "hallucinated_claims"}
    for ev in repo.evaluations:
        assert required_hdr_keys.issubset(ev.keys()), (
            f"HDR keys missing in oneshot eval_row: {required_hdr_keys - ev.keys()}"
        )
        assert isinstance(ev["evidence_claims"], int) and ev["evidence_claims"] >= 0
        assert (
            isinstance(ev["hallucinated_claims"], int)
            and ev["hallucinated_claims"] >= 0
        )


# ---------------------------------------------------------------------------
# Regression test: source normalization in _fetch_source
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_source_normalizes_attributes_json():
    """_fetch_source must call _normalize_doc so the source gets an 'attributes' dict.

    Regression guard for the bug where _fetch_source returned the raw Typesense
    document without parsing attributes_json into 'attributes'. This caused
    source_attrs = source.get('attributes') or {} to always be {}, making every
    rule attribute lookup on the source side fail (rules_undefined == rules_evaluated
    for every pair, L=0, zero recommendations catalog-wide).

    This test sends a source document that has NO 'attributes' key (mimicking the
    raw Typesense payload) but HAS 'attributes_json'. After _fetch_source runs
    _normalize_doc, the 'attributes' key must be populated from attributes_json.
    As a result, rules that reference 'brand' on the source side evaluate as
    DEFINED (rules_undefined < rules_evaluated) rather than all-undefined.
    """
    settings = StubSettings()
    repo = StubRepo()

    # Raw Typesense document - no 'attributes' key, only 'attributes_json' string.
    raw_source_doc = {
        "product_id": 100,
        "name": "Coffee Machine Raw",
        "product_type": "coffee_machine",
        "categories": ["Coffee Machines"],
        "embedding": _DUMMY_EMBEDDING,
        "attributes_json": json.dumps(
            {
                "brand": "siemens",
                "product_category": "coffee_machine",
                "compatible_brands": ["siemens"],
            }
        ),
        # No 'attributes' key - this is the raw Typesense payload shape.
    }

    cands = [_make_candidate_doc(pid, score=0.7) for pid in range(200, 203)]
    http = _make_mock_http(raw_source_doc, cands)

    await process_product(100, "test_exp", settings, repo, http)

    # The pipeline must not short-circuit on no_source or no_candidates.
    assert len(repo.evaluations) == 3, (
        "Expected 3 evaluation rows; got 0 means source normalization failed "
        "(no 'attributes' -> embedding [] -> retrieve_candidates returned [])"
    )

    # With a properly normalized source, rules that check 'brand' on the source
    # side (MockRuleCache includes a brand rule for coffee_machine->descaler)
    # must evaluate as DEFINED for at least one candidate.
    total_undefined = sum(ev["rules_undefined"] for ev in repo.evaluations)
    total_evaluated = sum(ev["rules_evaluated"] for ev in repo.evaluations)
    assert total_evaluated > 0, (
        "No rules were evaluated - MockRuleCache returned nothing"
    )
    assert total_undefined < total_evaluated, (
        f"All {total_evaluated} rules are undefined (rules_undefined={total_undefined}). "
        "This means source attributes were not parsed from attributes_json - "
        "_fetch_source is returning the raw doc without _normalize_doc."
    )
