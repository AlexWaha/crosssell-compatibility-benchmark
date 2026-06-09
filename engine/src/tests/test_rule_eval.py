"""Unit tests for app.services.compatibility.rule_eval.

Every assertion uses hand-verified, exact expected values. Comments show the
arithmetic so failures are diagnosable without a calculator.

Run: docker exec avtc_engine python -m pytest tests/test_rule_eval.py -v
"""

from __future__ import annotations

import math

from app.services.compatibility.rule_eval import (
    aggregate_logical,
    compute_context_code,
    evaluate_rule,
    normalize_value,
)

# ---------------------------------------------------------------------------
# normalize_value
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace():
    assert normalize_value("  Foo  ") == "foo"


def test_normalize_strips_unit_suffix_watts():
    # '1500 W' -> '1500' (unit stripped)
    assert normalize_value("1500 W") == "1500"


def test_normalize_strips_unit_suffix_kg():
    assert normalize_value("2.5 kg") == "2.5"


def test_normalize_lowercases():
    assert normalize_value("ESPRESSO") == "espresso"


def test_normalize_none():
    assert normalize_value(None) == ""


def test_normalize_integer():
    assert normalize_value(42) == "42"


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

_EM_RULE = {
    "type": "exact_match",
    "attribute_a": "socket_type",
    "attribute_b": "socket_type",
}


def test_exact_match_equal():
    # Normalized 'E14' == 'E14' -> l_k=1.0, passed
    attrs_i = {"socket_type": "E14"}
    attrs_j = {"socket_type": "E14"}
    l_k, status = evaluate_rule(_EM_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_exact_match_unequal():
    # 'E14' != 'E27' -> l_k=0.0, failed
    attrs_i = {"socket_type": "E14"}
    attrs_j = {"socket_type": "E27"}
    l_k, status = evaluate_rule(_EM_RULE, attrs_i, attrs_j)
    assert l_k == 0.0
    assert status == "failed"


def test_exact_match_missing_source():
    # source attr absent -> undefined
    l_k, status = evaluate_rule(_EM_RULE, {}, {"socket_type": "E14"})
    assert l_k == 0.0
    assert status == "undefined"


def test_exact_match_missing_target():
    # target attr absent -> undefined
    l_k, status = evaluate_rule(_EM_RULE, {"socket_type": "E14"}, {})
    assert l_k == 0.0
    assert status == "undefined"


def test_exact_match_normalisation():
    # '1500 W' vs '1500' -> after strip-units both become '1500' -> equal
    rule = {"type": "exact_match", "attribute_a": "wattage", "attribute_b": "wattage"}
    l_k, status = evaluate_rule(rule, {"wattage": "1500 W"}, {"wattage": "1500"})
    assert l_k == 1.0
    assert status == "passed"


# ---------------------------------------------------------------------------
# range_check
# ---------------------------------------------------------------------------

_RC_RULE = {
    "type": "range_check",
    "attribute_a": "max_wattage",
    "attribute_b": "wattage",
}


def test_range_check_within():
    # source max=2000, target=1500 -> 0 <= 1500 <= 2000 -> l_k=1.0, passed
    attrs_i = {"max_wattage": "2000"}
    attrs_j = {"wattage": "1500"}
    l_k, status = evaluate_rule(_RC_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_range_check_at_boundary():
    # target exactly at max -> still in range
    attrs_i = {"max_wattage": "1500"}
    attrs_j = {"wattage": "1500"}
    l_k, status = evaluate_rule(_RC_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_range_check_outside():
    # source max=1000, target=1500 -> 1500 > 1000 -> l_k=0.0, failed
    attrs_i = {"max_wattage": "1000"}
    attrs_j = {"wattage": "1500"}
    l_k, status = evaluate_rule(_RC_RULE, attrs_i, attrs_j)
    assert l_k == 0.0
    assert status == "failed"


def test_range_check_non_numeric_target():
    # target is a string that cannot be cast -> undefined
    attrs_i = {"max_wattage": "2000"}
    attrs_j = {"wattage": "high"}
    l_k, status = evaluate_rule(_RC_RULE, attrs_i, attrs_j)
    assert l_k == 0.0
    assert status == "undefined"


def test_range_check_missing_source():
    l_k, status = evaluate_rule(_RC_RULE, {}, {"wattage": "1500"})
    assert status == "undefined"


def test_range_check_explicit_min_max():
    # source '500-2000' format -> target 1500 is within [500, 2000]
    rule = {
        "type": "range_check",
        "attribute_a": "power_range",
        "attribute_b": "wattage",
    }
    attrs_i = {"power_range": "500-2000"}
    attrs_j = {"wattage": "1500"}
    l_k, status = evaluate_rule(rule, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_range_check_target_with_unit_suffix():
    # target value '1500 W' has a unit suffix that must be stripped before
    # numeric parsing. normalize_value removes the ' W', leaving '1500'.
    # source range '1000-2000': 1500 in [1000, 2000] -> l_k=1.0, passed
    rule = {
        "type": "range_check",
        "attribute_a": "power_range",
        "attribute_b": "wattage",
    }
    attrs_i = {"power_range": "1000-2000"}
    attrs_j = {"wattage": "1500 W"}
    l_k, status = evaluate_rule(rule, attrs_i, attrs_j)
    assert l_k == 1.0, f"Expected 1.0 (1500 W in [1000,2000]), got {l_k}"
    assert status == "passed"


# ---------------------------------------------------------------------------
# set_intersection
# ---------------------------------------------------------------------------

_SI_RULE = {
    "type": "set_intersection",
    "attribute_a": "supported_brands",
    "attribute_b": "supported_brands",
}


def test_set_intersection_full_overlap():
    # A={siemens}, B={siemens} -> |A & B|=1, min(|A|,|B|)=1 -> l_k=1.0
    attrs_i = {"supported_brands": ["siemens"]}
    attrs_j = {"supported_brands": ["siemens"]}
    l_k, status = evaluate_rule(_SI_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_set_intersection_partial_overlap():
    # A={siemens, bosch, neff}, B={siemens, miele}
    # |A & B| = 1 (siemens), min(3,2)=2 -> l_k = 1/2 = 0.5
    attrs_i = {"supported_brands": ["siemens", "bosch", "neff"]}
    attrs_j = {"supported_brands": ["siemens", "miele"]}
    l_k, status = evaluate_rule(_SI_RULE, attrs_i, attrs_j)
    assert abs(l_k - 0.5) < 1e-9
    assert status == "passed"


def test_set_intersection_no_overlap():
    # A={siemens}, B={miele} -> |A & B|=0 -> l_k=0.0, failed
    attrs_i = {"supported_brands": ["siemens"]}
    attrs_j = {"supported_brands": ["miele"]}
    l_k, status = evaluate_rule(_SI_RULE, attrs_i, attrs_j)
    assert l_k == 0.0
    assert status == "failed"


def test_set_intersection_missing_source():
    l_k, status = evaluate_rule(_SI_RULE, {}, {"supported_brands": ["siemens"]})
    assert status == "undefined"


def test_set_intersection_comma_separated():
    # Comma-separated string is split automatically
    attrs_i = {"supported_brands": "siemens,bosch"}
    attrs_j = {"supported_brands": "siemens,miele"}
    l_k, status = evaluate_rule(_SI_RULE, attrs_i, attrs_j)
    # A={siemens,bosch}, B={siemens,miele}: |A&B|=1, min(2,2)=2 -> 0.5
    assert abs(l_k - 0.5) < 1e-9
    assert status == "passed"


def test_set_intersection_exact_fraction_one_third():
    # A={siemens, bosch, neff}, B={neff, miele, philips}
    # intersection = {neff}, |A & B|=1, min(3,3)=3 -> l_k = 1/3
    attrs_i = {"supported_brands": ["siemens", "bosch", "neff"]}
    attrs_j = {"supported_brands": ["neff", "miele", "philips"]}
    l_k, status = evaluate_rule(_SI_RULE, attrs_i, attrs_j)
    assert abs(l_k - 1 / 3) < 1e-9, f"Expected 1/3, got {l_k}"
    assert status == "passed"


# ---------------------------------------------------------------------------
# regex_match
# ---------------------------------------------------------------------------

_RX_RULE = {
    "type": "regex_match",
    "attribute_a": "model_pattern",
    "attribute_b": "model_number",
}


def test_regex_match_hit():
    # pattern r'^EQ\.' matches 'EQ.6 plus' -> l_k=1.0, passed
    attrs_i = {"model_pattern": r"^EQ\."}
    attrs_j = {"model_number": "EQ.6 plus"}
    l_k, status = evaluate_rule(_RX_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


def test_regex_match_miss():
    # pattern does not match -> l_k=0.0, failed
    attrs_i = {"model_pattern": r"^EQ\."}
    attrs_j = {"model_number": "TK52001"}
    l_k, status = evaluate_rule(_RX_RULE, attrs_i, attrs_j)
    assert l_k == 0.0
    assert status == "failed"


def test_regex_match_missing_target():
    l_k, status = evaluate_rule(_RX_RULE, {"model_pattern": r"^EQ\."}, {})
    assert status == "undefined"


def test_regex_match_missing_source():
    l_k, status = evaluate_rule(_RX_RULE, {}, {"model_number": "EQ.6"})
    assert status == "undefined"


def test_regex_case_insensitive():
    # regex_match uses re.IGNORECASE
    attrs_i = {"model_pattern": "espresso"}
    attrs_j = {"model_number": "ESPRESSO 5000"}
    l_k, status = evaluate_rule(_RX_RULE, attrs_i, attrs_j)
    assert l_k == 1.0
    assert status == "passed"


# ---------------------------------------------------------------------------
# Unknown rule type
# ---------------------------------------------------------------------------


def test_unknown_rule_type_raises():
    import pytest

    with pytest.raises(ValueError, match="Unknown rule type"):
        evaluate_rule(
            {"type": "fuzzy_match", "attribute_a": "x", "attribute_b": "y"}, {}, {}
        )


# ---------------------------------------------------------------------------
# aggregate_logical (weighted_product)
# ---------------------------------------------------------------------------
#
# Hand-computed fixture:
#
#   Rule 1: l_k=0.8, weight=1.0, status='passed'
#   Rule 2: l_k=0.6, weight=0.7, status='passed'
#   Rule 3: l_k=1.0, weight=0.5, status='passed'
#
#   L = exp( 1.0*ln(0.8) + 0.7*ln(0.6) + 0.5*ln(1.0) )
#     = exp( 1.0*(-0.22314) + 0.7*(-0.51083) + 0.5*(0.0) )
#     = exp( -0.22314 + (-0.35758) + 0.0 )
#     = exp( -0.58072 )
#     = 0.55949  (6 decimal places)
#
# Verified with: math.exp(-0.22314355131 + 0.7*math.log(0.6)) = 0.55949...

_AGG_FIXTURE = [
    (0.8, 1.0, "passed"),
    (0.6, 0.7, "passed"),
    (1.0, 0.5, "passed"),
]

_AGG_EXPECTED_L = math.exp(
    1.0 * math.log(0.8) + 0.7 * math.log(0.6) + 0.5 * math.log(1.0)
)


def test_aggregate_weighted_product_value():
    """L matches hand-computed PROD l_k^w."""
    L, n_passed, n_failed, n_undefined = aggregate_logical(
        _AGG_FIXTURE, "weighted_product"
    )
    assert abs(L - _AGG_EXPECTED_L) < 1e-9
    assert n_passed == 3
    assert n_failed == 0
    assert n_undefined == 0


def test_aggregate_unweighted_product():
    # 'product' mode: L = PROD l_k = 0.8 * 0.6 * 1.0 = 0.48
    L, _, _, _ = aggregate_logical(_AGG_FIXTURE, "product")
    assert abs(L - 0.8 * 0.6 * 1.0) < 1e-9


def test_aggregate_with_undefined_rules():
    # One undefined rule is excluded from product
    # l1=0.8,w=1.0 defined; l2=0.6,w=0.7 defined; l3 undefined
    # L = exp(1.0*ln(0.8) + 0.7*ln(0.6)) = exp(-0.22314 - 0.35758) = exp(-0.58072)
    fixture = [(0.8, 1.0, "passed"), (0.6, 0.7, "passed"), (0.0, 0.5, "undefined")]
    expected = math.exp(1.0 * math.log(0.8) + 0.7 * math.log(0.6))
    L, n_passed, n_failed, n_undefined = aggregate_logical(fixture, "weighted_product")
    assert abs(L - expected) < 1e-9
    assert n_passed == 2
    assert n_failed == 0
    assert n_undefined == 1


def test_aggregate_all_undefined_returns_zero():
    # Conservative: if ALL rules are undefined -> L=0.0 (article requirement)
    fixture = [(0.0, 1.0, "undefined"), (0.0, 0.8, "undefined")]
    L, n_passed, n_failed, n_undefined = aggregate_logical(fixture)
    assert L == 0.0
    assert n_passed == 0
    assert n_failed == 0
    assert n_undefined == 2


def test_aggregate_zero_lk_collapses_product():
    # Any l_k=0.0 (non-undefined) collapses L to 0.0
    # Rule 1: l_k=0.8, passed; Rule 2: l_k=0.0, failed; Rule 3: l_k=0.9, passed
    fixture = [(0.8, 1.0, "passed"), (0.0, 0.8, "failed"), (0.9, 0.6, "passed")]
    L, n_passed, n_failed, n_undefined = aggregate_logical(fixture)
    assert L == 0.0
    assert n_passed == 2
    assert n_failed == 1
    assert n_undefined == 0


def test_aggregate_counts_failed_correctly():
    # A failed rule with l_k=0.0 collapses the entire product to 0.0.
    # Rule 1: l_k=0.8, w=1.0, passed
    # Rule 2: l_k=0.0, w=0.7, failed   <- zero collapses product
    # Rule 3: l_k=0.0, w=0.5, undefined <- excluded from product
    fixture = [(0.8, 1.0, "passed"), (0.0, 0.7, "failed"), (0.0, 0.5, "undefined")]
    L, n_passed, n_failed, n_undefined = aggregate_logical(fixture)
    # Any l_k=0.0 in the defined set collapses L to 0.0
    assert L == 0.0
    assert n_passed == 1
    assert n_failed == 1
    assert n_undefined == 1


# ---------------------------------------------------------------------------
# compute_context_code
# ---------------------------------------------------------------------------


def test_context_code_cleaning():
    rules: list[dict] = []
    assert compute_context_code(rules, "coffee_machine", "descaler") == "cleaning"


def test_context_code_filter():
    assert compute_context_code([], "coffee_machine", "water_filter") == "filter"


def test_context_code_appliance_fallback():
    # Both sides are appliances -> returns 'appliance'
    assert compute_context_code([], "coffee_machine", "electric_kettle") == "appliance"


def test_context_code_unknown_types():
    # Unknown types -> falls back to 'accessory'
    assert compute_context_code([], "gadget_x", "gadget_y") == "accessory"
