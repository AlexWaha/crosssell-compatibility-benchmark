"""Unit tests for app.services.normalization (normalizer + prompt).

All tests run at $0 / mock - no LLM, no DB, no network.
Covers: MockNormalizer UPS shape, source_hash determinism, normalize_product metrics,
and parse_normalization validation.

Run: docker exec avtc_engine python -m pytest tests/test_normalizer.py -v
"""

from __future__ import annotations

import asyncio


from app.services.normalization.normalizer import (
    MockNormalizer,
    compute_source_hash,
    normalize_product,
)
from app.services.normalization.prompt import PROMPT_VERSION, parse_normalization

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PRODUCT = {
    "product_id": 999,
    "name": "Siemens EQ.6 Plus Coffee Machine",
    "description": "Fully automatic espresso machine with 1500W power and 15 bar pressure.",
    "brand": "Siemens",
    "product_type": "",
    "status": 1,
}

_RAW_ATTRS = {
    "Leistung": "1500 W",
    "Druck": "15 bar",
    "Farbe": "schwarz",
}

_CATEGORIES = ["Coffee Machines", "Espresso"]


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# MockNormalizer
# ---------------------------------------------------------------------------


def test_mock_normalizer_returns_valid_ups():
    """MockNormalizer must produce a dict that passes parse_normalization."""
    mock = MockNormalizer(model="mock")
    ups, ti, to = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))

    assert ups is not None, "MockNormalizer returned None"
    validated = parse_normalization(ups)
    assert validated is not None, f"parse_normalization rejected mock output: {ups}"


def test_mock_normalizer_has_all_four_fields():
    """MockNormalizer output must contain all four required UPS fields."""
    mock = MockNormalizer(model="mock")
    ups, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))

    assert "product_type" in ups
    assert "normalized_json" in ups
    assert "compatibility_tags" in ups
    assert "embedding_text" in ups


def test_mock_normalizer_zero_tokens():
    """MockNormalizer must return zero tokens (no LLM call)."""
    mock = MockNormalizer(model="mock")
    _, ti, to = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    assert ti == 0
    assert to == 0


def test_mock_normalizer_product_type_plausible():
    """MockNormalizer derives a plausible product_type from the product name."""
    mock = MockNormalizer(model="mock")
    ups, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    # "Coffee Machine" in name -> should resolve to coffee_machine
    assert ups["product_type"] == "coffee_machine"


def test_mock_normalizer_tags_nonempty():
    """MockNormalizer must produce at least one compatibility tag."""
    mock = MockNormalizer(model="mock")
    ups, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    assert isinstance(ups["compatibility_tags"], list)
    assert len(ups["compatibility_tags"]) >= 1


def test_mock_normalizer_embedding_text_nonempty():
    """MockNormalizer embedding_text must be a non-empty string."""
    mock = MockNormalizer(model="mock")
    ups, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    assert isinstance(ups["embedding_text"], str)
    assert len(ups["embedding_text"]) > 0


def test_mock_normalizer_embedding_text_max_300():
    """MockNormalizer embedding_text must not exceed 300 characters."""
    mock = MockNormalizer(model="mock")
    ups, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    assert len(ups["embedding_text"]) <= 300


def test_mock_normalizer_deterministic():
    """Two calls with identical inputs must produce identical output."""
    mock = MockNormalizer(model="mock")
    ups1, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    ups2, _, _ = _run(mock.normalize(_PRODUCT, _RAW_ATTRS, _CATEGORIES))
    assert ups1 == ups2


# ---------------------------------------------------------------------------
# compute_source_hash
# ---------------------------------------------------------------------------


def test_source_hash_deterministic():
    """Same inputs always produce the same hash."""
    h1 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock", PROMPT_VERSION)
    h2 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock", PROMPT_VERSION)
    assert h1 == h2


def test_source_hash_length():
    """Source hash must be a 64-character hex string (SHA-256)."""
    h = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock", PROMPT_VERSION)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_source_hash_changes_on_prompt_version():
    """Changing prompt version must produce a different hash (cache invalidation)."""
    h1 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock", "norm_v1")
    h2 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock", "norm_v2")
    assert h1 != h2


def test_source_hash_changes_on_model():
    """Changing model name must produce a different hash."""
    h1 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock")
    h2 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "gpt-4o")
    assert h1 != h2


def test_source_hash_changes_on_name():
    """Changing product name must produce a different hash."""
    product_alt = {**_PRODUCT, "name": "Different Product Name"}
    h1 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock")
    h2 = compute_source_hash(product_alt, _RAW_ATTRS, _CATEGORIES, "mock")
    assert h1 != h2


def test_source_hash_changes_on_attrs():
    """Adding a new attribute must produce a different hash."""
    extra_attrs = {**_RAW_ATTRS, "new_attr": "new_value"}
    h1 = compute_source_hash(_PRODUCT, _RAW_ATTRS, _CATEGORIES, "mock")
    h2 = compute_source_hash(_PRODUCT, extra_attrs, _CATEGORIES, "mock")
    assert h1 != h2


def test_source_hash_stable_across_attr_order():
    """Source hash must be stable regardless of attribute dict insertion order."""
    attrs_a = {"Leistung": "1500 W", "Druck": "15 bar"}
    attrs_b = {"Druck": "15 bar", "Leistung": "1500 W"}
    h1 = compute_source_hash(_PRODUCT, attrs_a, _CATEGORIES, "mock")
    h2 = compute_source_hash(_PRODUCT, attrs_b, _CATEGORIES, "mock")
    assert h1 == h2


# ---------------------------------------------------------------------------
# normalize_product metrics
# ---------------------------------------------------------------------------


def test_normalize_product_metrics_shape():
    """normalize_product must return a dict with 'metrics' containing all expected keys."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(
            mock,
            _PRODUCT,
            _RAW_ATTRS,
            _CATEGORIES,
            model="mock",
            prompt_version=PROMPT_VERSION,
        )
    )

    assert "metrics" in result
    m = result["metrics"]
    required = {
        "attrs_before",
        "attrs_after",
        "attrs_missing_before",
        "filled",
        "fill_rate",
        "llm_calls",
        "tokens_input",
        "tokens_output",
    }
    assert required.issubset(m.keys()), f"Missing metric keys: {required - m.keys()}"


def test_normalize_product_attrs_before_count():
    """attrs_before must equal the number of raw attributes provided."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    assert result["metrics"]["attrs_before"] == len(_RAW_ATTRS)


def test_normalize_product_fill_counts_new_attrs():
    """filled must count attributes present in normalized output but absent in raw."""
    # Product with zero raw attributes - all normalized attrs count as filled
    mock = MockNormalizer(model="mock")
    result = _run(normalize_product(mock, _PRODUCT, {}, _CATEGORIES, model="mock"))
    m = result["metrics"]
    # With empty raw_attrs, all normalized keys should count as filled
    assert m["filled"] == m["attrs_after"]


def test_normalize_product_fill_rate_range():
    """fill_rate must be a float in [0.0, 1.0] and consistent with filled/attrs_before."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    m = result["metrics"]
    assert isinstance(m["fill_rate"], float)
    assert 0.0 <= m["fill_rate"] <= 1.0
    # fill_rate = min(filled / max(attrs_before, 1), 1.0) when attrs_before > 0
    # the cap only fires when attrs_before == 0; here attrs_before == 3 so the
    # formula reduces to filled / attrs_before without triggering the cap.
    expected = m["filled"] / max(m["attrs_before"], 1)
    assert abs(m["fill_rate"] - expected) < 1e-9


def test_normalize_product_mock_zero_tokens():
    """MockNormalizer produces zero tokens in normalize_product metrics."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    m = result["metrics"]
    assert m["tokens_input"] == 0
    assert m["tokens_output"] == 0


def test_normalize_product_llm_calls_is_one_on_success():
    """llm_calls must be 1 when normalization succeeds (even for mock)."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    assert result["metrics"]["llm_calls"] == 1


def test_normalize_product_source_hash_present():
    """normalize_product must include a non-empty source_hash."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    assert "source_hash" in result
    assert len(result["source_hash"]) == 64


def test_normalize_product_success_flag():
    """normalize_product must set success=True when normalization succeeds."""
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, _RAW_ATTRS, _CATEGORIES, model="mock")
    )
    assert result["success"] is True


def test_normalize_product_missing_attrs_counted():
    """attrs_missing_before counts raw attrs with empty/None values."""
    attrs_with_empty = {"Leistung": "1500 W", "Farbe": "", "Druck": "15 bar"}
    mock = MockNormalizer(model="mock")
    result = _run(
        normalize_product(mock, _PRODUCT, attrs_with_empty, _CATEGORIES, model="mock")
    )
    # 'Farbe' has empty value -> missing_before = 1
    assert result["metrics"]["attrs_missing_before"] == 1


# ---------------------------------------------------------------------------
# parse_normalization
# ---------------------------------------------------------------------------


def test_parse_normalization_valid_dict():
    """parse_normalization accepts a valid dict with all required fields."""
    ups = {
        "product_type": "coffee_machine",
        "normalized_json": {"max_power_w": "1500"},
        "compatibility_tags": ["espresso", "siemens"],
        "embedding_text": "Siemens EQ.6 coffee machine 1500w",
    }
    result = parse_normalization(ups)
    assert result is not None
    assert result["product_type"] == "coffee_machine"


def test_parse_normalization_normalizes_type_case():
    """parse_normalization lowercases and snake_cases the product_type."""
    ups = {
        "product_type": "Coffee Machine",
        "normalized_json": {},
        "compatibility_tags": [],
        "embedding_text": "test",
    }
    result = parse_normalization(ups)
    assert result is not None
    assert result["product_type"] == "coffee_machine"


def test_parse_normalization_rejects_missing_field():
    """parse_normalization returns None if a required field is missing."""
    ups = {
        "product_type": "coffee_machine",
        "normalized_json": {},
        # missing compatibility_tags and embedding_text
    }
    assert parse_normalization(ups) is None


def test_parse_normalization_rejects_none():
    """parse_normalization returns None for None input."""
    assert parse_normalization(None) is None


def test_parse_normalization_rejects_empty_product_type():
    """parse_normalization returns None when product_type is empty."""
    ups = {
        "product_type": "",
        "normalized_json": {},
        "compatibility_tags": [],
        "embedding_text": "test",
    }
    assert parse_normalization(ups) is None


def test_parse_normalization_accepts_json_string():
    """parse_normalization accepts a JSON string as input."""
    import json

    ups = {
        "product_type": "smartphone",
        "normalized_json": {"ram_gb": "8"},
        "compatibility_tags": ["android"],
        "embedding_text": "smartphone android 8gb ram",
    }
    result = parse_normalization(json.dumps(ups))
    assert result is not None
    assert result["product_type"] == "smartphone"


def test_parse_normalization_rejects_invalid_json_string():
    """parse_normalization returns None on unparseable JSON string."""
    assert parse_normalization("not json {{{") is None


def test_parse_normalization_tags_lowercased():
    """parse_normalization lowercases all compatibility tags."""
    ups = {
        "product_type": "laptop",
        "normalized_json": {},
        "compatibility_tags": ["Windows", "USB-C"],
        "embedding_text": "laptop windows usb-c",
    }
    result = parse_normalization(ups)
    assert result is not None
    assert result["compatibility_tags"] == ["windows", "usb-c"]


# ---------------------------------------------------------------------------
# JITOE fill_rate integrity tests (article-grade metric)
# ---------------------------------------------------------------------------


class _StubNormalizer:
    """Stub normalizer that returns a fixed normalized_json for testing fill semantics."""

    def __init__(self, normalized_json: dict) -> None:
        self._nj = normalized_json

    async def normalize(
        self,
        product: dict,
        raw_attrs: dict[str, str],
        categories: list[str],
    ) -> tuple[dict, int, int]:
        ups = {
            "product_type": "product",
            "normalized_json": self._nj,
            "compatibility_tags": ["product"],
            "embedding_text": "stub",
        }
        return ups, 0, 0


def test_fill_count_partial_overlap():
    """filled counts only genuinely new attrs, not raw-key renames or pass-throughs.

    Scenario:
      raw_attrs = {"Leistung": "1500 W", "Druck": "15 bar"}
      normalized output = {
          "leistung": "1500 W",   # pass-through (key matches raw canonical + value matches)
          "druck": "15 bar",      # pass-through (same)
          "color": "red",         # genuinely new - key absent from raw, value absent from raw
      }
    Expected: filled == 1 (only "color" is a genuine LLM inference).
    """
    raw_attrs = {"Leistung": "1500 W", "Druck": "15 bar"}
    normalized_json = {
        "leistung": "1500 W",  # renamed key, same value -> NOT filled
        "druck": "15 bar",  # renamed key, same value -> NOT filled
        "color": "red",  # new key, new value -> FILLED
    }
    stub = _StubNormalizer(normalized_json)
    result = _run(
        normalize_product(stub, _PRODUCT, raw_attrs, _CATEGORIES, model="mock")
    )
    m = result["metrics"]
    assert m["filled"] == 1, (
        f"Expected filled=1 (only 'color' is genuinely new), got filled={m['filled']}. "
        f"normalized_json={normalized_json}, raw_attrs={raw_attrs}"
    )


def test_fill_rate_capped_when_no_raw_attrs():
    """fill_rate is capped at 1.0 even when attrs_before == 0.

    A product with zero raw attributes causes attrs_before=0, so the denominator
    clamps to 1.  If the LLM infers N attributes the raw filled count is N, but
    fill_rate must not exceed 1.0 (the cap prevents misleading inflation of the
    JITOE metric in the article).
    """
    # MockNormalizer with empty raw_attrs still produces normalized_json entries
    # derived from the product name/type heuristics (none, actually - it only
    # iterates raw_attrs for normalized keys).  Use a stub that returns a known
    # non-empty normalized_json to guarantee filled > 1.
    normalized_json = {
        "color": "black",
        "material": "stainless_steel",
        "capacity_l": "1.7",
    }
    stub = _StubNormalizer(normalized_json)
    result = _run(normalize_product(stub, _PRODUCT, {}, _CATEGORIES, model="mock"))
    m = result["metrics"]
    # All 3 attrs are genuinely new (no raw to match against)
    assert m["filled"] == 3, f"Expected raw filled count=3, got {m['filled']}"
    # fill_rate must be capped at 1.0, not 3.0
    assert m["fill_rate"] == 1.0, (
        f"Expected fill_rate=1.0 (capped), got fill_rate={m['fill_rate']}. "
        "fill_rate > 1.0 would inflate the JITOE metric in the article."
    )
    assert m["attrs_before"] == 0


# ---------------------------------------------------------------------------
# Contract tests: load_raw_product output shape through the normalizer pipeline
#
# These tests use a fixture that exactly mirrors the dict returned by
# CatalogRepository.load_raw_product so that any schema mismatch between the
# repo and the normalizer is caught at $0. This is the class of bug that the
# original mock tests missed (those tests passed a hand-crafted dict while the
# repo previously returned a list, or the CLI failed to create a real client).
# ---------------------------------------------------------------------------


# Exact output shape of CatalogRepository.load_raw_product
_REPO_OUTPUT = {
    "product": {
        "product_id": 1,
        "name": "Kids School Backpack Dinosaur",
        "description": "Lightweight backpack for kids age 3-7 with dinosaur print.",
        "brand": "FunKids",
        "product_type": "",
        "status": 1,
    },
    "categories": ["School Bags", "Kids & Toys"],
    # raw_attrs is a dict {attribute_name: attribute_value} - same as DB query result
    "raw_attrs": {
        "Age Group": "preschool",
        "Cartoon Style": "Yes",
        "Dimensions (HxWxD)": "35x32x14 cm",
        "Weight": "0.45 kg",
        "Material": "Polyester",
    },
}


def test_contract_repo_output_shape_through_source_hash():
    """compute_source_hash must accept the exact dict shape that load_raw_product returns.

    This test catches any mismatch between the repo's raw_attrs format and what
    compute_source_hash expects. If load_raw_product ever returns a list instead of
    a dict, this test fails with AttributeError: 'list' has no attribute 'items'.
    """
    product = _REPO_OUTPUT["product"]
    raw_attrs = _REPO_OUTPUT["raw_attrs"]
    categories = _REPO_OUTPUT["categories"]

    # Must not raise - must return a 64-char hex string
    h = compute_source_hash(product, raw_attrs, categories, "mock")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_contract_repo_output_shape_through_normalize_product():
    """normalize_product must process the exact load_raw_product output without error.

    This end-to-end contract test exercises: compute_source_hash + MockNormalizer.normalize
    + metric calculation. Any 'AttributeError on list' failure surfaces here before any
    real LLM spend.
    """
    product = _REPO_OUTPUT["product"]
    raw_attrs = _REPO_OUTPUT["raw_attrs"]
    categories = _REPO_OUTPUT["categories"]

    mock = MockNormalizer(model="mock")
    result = _run(normalize_product(mock, product, raw_attrs, categories, model="mock"))

    assert result["success"] is True, f"normalize_product failed: {result}"
    assert result["product_type"] is not None
    assert result["embedding_text"] is not None
    assert len(result["source_hash"]) == 64
    assert result["metrics"]["attrs_before"] == len(raw_attrs)


def test_contract_list_raw_attrs_tolerated_by_compute_source_hash():
    """compute_source_hash must not crash if a list of {name, value} dicts is passed.

    This guards the defensive conversion added to compute_source_hash. It does NOT
    mean the list format is the canonical contract - the canonical contract is a dict.
    Callers that pass a list get a converted result, not a crash.
    """
    product = _REPO_OUTPUT["product"]
    # Simulate the malformed list format that would have caused the original bug
    raw_attrs_list = [
        {"name": "Age Group", "value": "preschool"},
        {"name": "Weight", "value": "0.45 kg"},
    ]
    categories = _REPO_OUTPUT["categories"]

    # Must not raise AttributeError - must return a valid hash
    h = compute_source_hash(product, raw_attrs_list, categories, "mock")
    assert len(h) == 64

    # The hash from a correctly-converted list must equal the hash from the equivalent dict
    raw_attrs_dict = {"Age Group": "preschool", "Weight": "0.45 kg"}
    h_dict = compute_source_hash(product, raw_attrs_dict, categories, "mock")
    assert h == h_dict, (
        "Hash from list-format raw_attrs must equal hash from equivalent dict "
        "so that the defensive guard does not create a new cache-miss on every run."
    )


def test_contract_mock_normalizer_model_used_is_mock():
    """MockNormalizer must record model_used='mock' so mock rows are distinguishable.

    The normalize.py CLI writes model_used from _model_name(settings), which returns
    'mock' when LLM_MODE='mock'. This test verifies that MockNormalizer is always
    initialized with model='mock' by get_normalizer in mock mode (the factory default).
    A different model string would write a non-mock model_used value and would also
    produce different source_hashes, masking whether a row was written by mock or real.
    """
    from app.services.normalization.normalizer import get_normalizer

    class _FakeCfg:
        LLM_MODE = "mock"
        PRIMARY_MODEL = "gpt-5-nano"

    n = get_normalizer(_FakeCfg())
    assert isinstance(n, MockNormalizer)
    # The model field on MockNormalizer must be 'mock' - it feeds the source_hash
    # and is what _model_name() returns for LLM_MODE='mock'
    assert n._model == "mock", (
        f"MockNormalizer._model must be 'mock', got '{n._model}'. "
        "If this differs from what _model_name() returns, mock and real hashes "
        "collide and a mode switch will not trigger reprocessing."
    )
