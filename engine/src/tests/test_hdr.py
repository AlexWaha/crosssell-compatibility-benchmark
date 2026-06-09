"""Tests for HDR (Hallucination Detection and Rejection) / evidence gate.

Verifies that rules referencing attributes absent from the product vocabulary
are flagged as hallucinated and excluded from L computation.

Run: docker exec avtc_engine python -m pytest tests/test_hdr.py -v
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.compatibility.cache import MockRuleCache
from app.services.compatibility.ontology import (
    _apply_hdr_gate,
    _validate_rules,
    get_rules,
)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Vocabulary for coffee_machine products (real attributes that actually appear)
# ---------------------------------------------------------------------------

COFFEE_MACHINE_VOCAB = {
    "brand",
    "product_category",
    "compatible_brands",
    "max_wattage",
    "wattage",
    "water_connection_type",
    "heating_element_type",
    "appliance_category",
}

# ---------------------------------------------------------------------------
# Fabricated rules with a hallucinated attribute
# ---------------------------------------------------------------------------

RULES_WITH_HALLUCINATED = [
    {
        "id": "rule_1",
        "type": "exact_match",
        "attribute_a": "brand",  # REAL: in COFFEE_MACHINE_VOCAB
        "attribute_b": "brand",
        "weight": 0.9,
        "description": "Brand match (real attribute)",
    },
    {
        "id": "rule_2",
        "type": "exact_match",
        "attribute_a": "socket_type",  # HALLUCINATED: NOT in COFFEE_MACHINE_VOCAB
        "attribute_b": "socket_type",
        "weight": 0.8,
        "description": "Socket type match (hallucinated attribute)",
    },
    {
        "id": "rule_3",
        "type": "set_intersection",
        "attribute_a": "compatible_brands",  # REAL
        "attribute_b": "compatible_brands",
        "weight": 0.7,
        "description": "Compatible brands overlap (real attribute)",
    },
]


def test_hdr_drops_hallucinated_rule():
    """Rule referencing socket_type (absent from coffee_machine vocab) is dropped."""
    filtered, evidence_claims, hallucinated_claims = _apply_hdr_gate(
        RULES_WITH_HALLUCINATED,
        vocab_a=COFFEE_MACHINE_VOCAB,
        vocab_b=COFFEE_MACHINE_VOCAB,
    )
    # Only 2 of 3 rules survive the gate
    assert len(filtered) == 2, f"Expected 2 rules after HDR, got {len(filtered)}"
    assert hallucinated_claims == 1, (
        f"Expected 1 hallucinated claim, got {hallucinated_claims}"
    )
    assert evidence_claims == 2, f"Expected 2 evidence claims, got {evidence_claims}"


def test_hdr_dropped_rule_is_socket_type():
    """The dropped rule is specifically the one referencing socket_type."""
    filtered, _, _ = _apply_hdr_gate(
        RULES_WITH_HALLUCINATED,
        vocab_a=COFFEE_MACHINE_VOCAB,
        vocab_b=COFFEE_MACHINE_VOCAB,
    )
    surviving_ids = {r["id"] for r in filtered}
    assert "rule_2" not in surviving_ids, (
        "rule_2 (socket_type, hallucinated) must be excluded"
    )
    assert "rule_1" in surviving_ids, "rule_1 (brand, real) must survive"
    assert "rule_3" in surviving_ids, "rule_3 (compatible_brands, real) must survive"


def test_hdr_no_vocab_skips_gate():
    """When vocab is None on both sides, no rules are dropped."""
    filtered, evidence_claims, hallucinated_claims = _apply_hdr_gate(
        RULES_WITH_HALLUCINATED,
        vocab_a=None,
        vocab_b=None,
    )
    assert len(filtered) == 3
    assert hallucinated_claims == 0
    assert evidence_claims == 3


def test_hdr_partial_vocab_a_only():
    """When only vocab_a is supplied, only attribute_a is gated."""
    filtered, _, hallucinated = _apply_hdr_gate(
        RULES_WITH_HALLUCINATED,
        vocab_a=COFFEE_MACHINE_VOCAB,
        vocab_b=None,  # attribute_b not gated
    )
    # rule_2 attribute_a='socket_type' is NOT in vocab_a -> dropped
    assert len(filtered) == 2
    assert hallucinated == 1


def test_hdr_partial_vocab_b_only():
    """When only vocab_b is supplied, only attribute_b is gated."""
    # Use a vocab that DOES contain socket_type for side_b (so rule_2 survives)
    vocab_b_with_socket = COFFEE_MACHINE_VOCAB | {"socket_type"}
    filtered, _, hallucinated = _apply_hdr_gate(
        RULES_WITH_HALLUCINATED,
        vocab_a=None,
        vocab_b=vocab_b_with_socket,
    )
    # All rules survive because vocab_b accepts socket_type and vocab_a is not gated
    assert len(filtered) == 3
    assert hallucinated == 0


def test_hdr_all_rules_hallucinated_returns_empty():
    """When all rules reference absent attributes, result is empty list."""
    hallucinated_rules = [
        {
            "id": "rule_1",
            "type": "exact_match",
            "attribute_a": "socket_type",
            "attribute_b": "socket_type",
            "weight": 1.0,
            "description": "Hallucinated",
        },
        {
            "id": "rule_2",
            "type": "range_check",
            "attribute_a": "voltage_range",
            "attribute_b": "voltage",
            "weight": 0.8,
            "description": "Also hallucinated",
        },
    ]
    filtered, evidence_claims, hallucinated_claims = _apply_hdr_gate(
        hallucinated_rules,
        vocab_a=COFFEE_MACHINE_VOCAB,
        vocab_b=COFFEE_MACHINE_VOCAB,
    )
    assert len(filtered) == 0
    assert hallucinated_claims == 2
    assert evidence_claims == 0


def test_hdr_enabled_via_get_rules():
    """get_rules with hdr_enabled=True and vocab drops the hallucinated rule."""

    class FixedRulesLLM:
        """Returns the hallucinated rule set regardless of type pair."""

        _model_id = "fixed_rules_test"

        async def generate(self, system, user, fmt):
            return {"rules": RULES_WITH_HALLUCINATED}, 10, 20

    cache = MockRuleCache()
    llm = FixedRulesLLM()

    rules, evidence, hallucinated, was_miss = _run(
        get_rules(
            "coffee_machine",
            "coffee_machine",
            llm,
            cache,
            hdr_enabled=True,
            vocab_a=COFFEE_MACHINE_VOCAB,
            vocab_b=COFFEE_MACHINE_VOCAB,
        )
    )

    assert len(rules) == 2, f"Expected 2 rules after HDR gate, got {len(rules)}"
    assert hallucinated == 1
    surviving_ids = {r["id"] for r in rules}
    assert "rule_2" not in surviving_ids


def test_hdr_disabled_via_get_rules():
    """get_rules with hdr_enabled=False returns all rules including hallucinated."""

    class FixedRulesLLM:
        _model_id = "fixed_rules_test"

        async def generate(self, system, user, fmt):
            return {"rules": RULES_WITH_HALLUCINATED}, 10, 20

    cache = MockRuleCache()
    llm = FixedRulesLLM()

    rules, _, hallucinated, was_miss = _run(
        get_rules(
            "coffee_machine",
            "coffee_machine",
            llm,
            cache,
            hdr_enabled=False,
        )
    )
    assert len(rules) == 3
    assert hallucinated == 0


# ---------------------------------------------------------------------------
# _validate_rules: malformed rules are dropped, not crash
# ---------------------------------------------------------------------------


def test_validate_rules_drops_unknown_type():
    """A rule with an unknown type is dropped (warning), valid rules survive.

    Input: 3 rules where rule_2 has type 'fuzzy_match' (not in _VALID_RULE_TYPES).
    Expected: 2 valid rules returned, no ValueError raised.
    """
    raw = [
        {
            "id": "rule_1",
            "type": "exact_match",
            "attribute_a": "brand",
            "attribute_b": "brand",
            "weight": 0.9,
            "description": "Brand match",
        },
        {
            "id": "rule_2",
            "type": "fuzzy_match",  # unknown type - must be dropped
            "attribute_a": "model",
            "attribute_b": "model",
            "weight": 0.5,
            "description": "Fuzzy match (invalid type)",
        },
        {
            "id": "rule_3",
            "type": "set_intersection",
            "attribute_a": "compatible_brands",
            "attribute_b": "compatible_brands",
            "weight": 0.7,
            "description": "Compatible brands overlap",
        },
    ]
    validated = _validate_rules(raw)
    assert len(validated) == 2, f"Expected 2 valid rules, got {len(validated)}"
    surviving_ids = {r["id"] for r in validated}
    assert "rule_2" not in surviving_ids, "rule_2 (unknown type) must be dropped"
    assert "rule_1" in surviving_ids
    assert "rule_3" in surviving_ids


def test_validate_rules_drops_missing_key():
    """A rule missing a required key is dropped (warning), valid rules survive."""
    raw = [
        {
            "id": "rule_1",
            "type": "exact_match",
            "attribute_a": "brand",
            "attribute_b": "brand",
            "weight": 0.9,
            "description": "Good rule",
        },
        {
            # missing 'description' key - must be dropped
            "id": "rule_2",
            "type": "exact_match",
            "attribute_a": "voltage",
            "attribute_b": "voltage",
            "weight": 0.8,
        },
        {
            "id": "rule_3",
            "type": "range_check",
            "attribute_a": "max_wattage",
            "attribute_b": "wattage",
            "weight": 0.7,
            "description": "Power range check",
        },
    ]
    validated = _validate_rules(raw)
    assert len(validated) == 2
    surviving_ids = {r["id"] for r in validated}
    assert "rule_2" not in surviving_ids
    assert "rule_1" in surviving_ids
    assert "rule_3" in surviving_ids


def test_validate_rules_raises_if_too_few_valid():
    """ValueError is raised only when surviving valid rule count < _MIN_RULES."""
    raw = [
        {
            "id": "rule_1",
            "type": "unknown_type",  # dropped
            "attribute_a": "x",
            "attribute_b": "y",
            "weight": 0.5,
            "description": "bad",
        },
        {
            "id": "rule_2",
            "type": "also_invalid",  # dropped
            "attribute_a": "a",
            "attribute_b": "b",
            "weight": 0.3,
            "description": "also bad",
        },
    ]
    with pytest.raises(ValueError, match="valid rules"):
        _validate_rules(raw)


# ---------------------------------------------------------------------------
# get_rules was_cache_miss flag: True on LLM call, False on cache hit
# ---------------------------------------------------------------------------


def test_get_rules_cache_miss_flag_true_on_first_call():
    """get_rules returns was_cache_miss=True when LLM generate() is called."""
    from app.services.compatibility.cache import MockRuleCache
    from app.services.compatibility.ontology import MockRuleGen

    cache = MockRuleCache()
    llm = MockRuleGen()

    _, _, _, was_miss = _run(
        get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False)
    )
    assert was_miss is True, (
        "First get_rules call (cache miss) must return was_cache_miss=True"
    )


def test_get_rules_cache_miss_flag_false_on_cache_hit():
    """get_rules returns was_cache_miss=False on cache hit (no LLM call)."""
    from app.services.compatibility.cache import MockRuleCache
    from app.services.compatibility.ontology import MockRuleGen

    cache = MockRuleCache()
    llm = MockRuleGen()

    # Populate cache on first call.
    _run(get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False))

    # Second call: cache hit.
    _, _, _, was_miss = _run(
        get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False)
    )
    assert was_miss is False, (
        "Second get_rules call (cache hit) must return was_cache_miss=False"
    )
