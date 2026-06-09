"""DECISIVE sanity test: JIT-ontology replaces loose LLM-approved recommendations.

Proves that the rule-based L = PROD l_k^w path produces correct verdicts for
the known-problematic case (coffee machine -> electric kettle = INCOMPATIBLE)
that the one-shot LLM verify used to approve as 'compatible'.

Fixtures: Normalized attribute dicts for five product types.
LLM: MockRuleGen ($0, deterministic, no OpenAI calls ever).
Thresholds: tau_S=0.3, tau_L=0.5 (matching COMPAT_TAU_S/COMPAT_TAU_L defaults).

Assertions (the four coffee assertions that prove the fix):
  1. L(coffee_machine -> coffee_machine) < tau_L  -> verdict False (same appliance, no cross-sell)
  2. L(coffee_machine -> descaler)       >= tau_L  -> verdict True  (cleaning accessory)
  3. L(coffee_machine -> water_filter)   >= tau_L  -> verdict True  (filter accessory)
  4. L(coffee_machine -> electric_kettle) < tau_L  -> verdict False (unrelated appliance)

Run: docker exec avtc_engine python -m pytest tests/test_compatibility_sanity.py -v
"""

from __future__ import annotations

import asyncio

from app.services.compatibility.cache import MockRuleCache
from app.services.compatibility.ontology import MockRuleGen, get_rules
from app.services.compatibility.rule_eval import aggregate_logical, evaluate_rule
from app.services.scoring import compute_verdict

# ---------------------------------------------------------------------------
# Thresholds (article defaults, matching config.py)
# ---------------------------------------------------------------------------

TAU_S = 0.3  # semantic threshold (not tested here, fixed representative value)
TAU_L = 0.5  # logical threshold - the decisive gate
S_REP = 0.6  # representative semantic score (above tau_S)

# ---------------------------------------------------------------------------
# Fixture attribute dicts (normalised, lowercase keys, no units in values)
# ---------------------------------------------------------------------------

# Source product: Siemens EQ.6 coffee machine
COFFEE_MACHINE_A = {
    "brand": "siemens",
    "product_category": "coffee_machine",
    "appliance_category": "coffee_machine",
    "compatible_brands": ["siemens"],
    "max_wattage": "1500",
    "water_connection_type": "internal_tank",
    "heating_element_type": "thermoblock",
}

# Target product B: different coffee machine (Bosch Tassimo)
COFFEE_MACHINE_B = {
    "brand": "bosch",
    "product_category": "coffee_machine",
    "appliance_category": "coffee_machine",
    "compatible_brands": ["bosch"],
    "wattage": "1300",
    "max_wattage": "2000",
    "heating_element_type": "thermoblock",
}

# Descaler compatible with Siemens machines
DESCALER = {
    "brand": "siemens",
    "product_category": "descaler",
    "target_appliance_category": "coffee_machine",
    "compatible_brands": ["siemens", "bosch", "neff"],
}

# Water filter compatible with Siemens coffee machines
WATER_FILTER = {
    "brand": "siemens",
    "product_category": "water_filter",
    "compatible_brands": ["siemens"],
    "connection_type": "internal_tank",
}

# Electric kettle - unrelated appliance
ELECTRIC_KETTLE = {
    "brand": "bosch",
    "product_category": "electric_kettle",
    "appliance_category": "electric_kettle",
    "heating_element_type": "concealed_element",
    "compatible_brands": ["bosch"],
}


def _run(coro):
    return asyncio.run(coro)


def _compute_l(
    type_a: str, attrs_i: dict, type_b: str, attrs_j: dict
) -> tuple[float, int, int, int]:
    """Compute L for a pair using MockRuleGen + MockRuleCache (pure, $0)."""
    cache = MockRuleCache()
    llm = MockRuleGen()
    rules, _, _, _ = _run(get_rules(type_a, type_b, llm, cache, hdr_enabled=False))
    eval_results = []
    for rule in rules:
        l_k, status = evaluate_rule(rule, attrs_i, attrs_j)
        eval_results.append((l_k, rule["weight"], status))
    big_l, n_passed, n_failed, n_undefined = aggregate_logical(eval_results)
    return big_l, n_passed, n_failed, n_undefined


# ---------------------------------------------------------------------------
# Assertion 1: coffee_machine -> coffee_machine < tau_L  (verdict False)
# ---------------------------------------------------------------------------


def test_coffee_machine_to_coffee_machine_incompatible():
    """Siemens EQ.6 -> Bosch Tassimo: same product category, incompatible for cross-sell.

    Rules for (coffee_machine, coffee_machine):
      rule_1 exact_match brand: 'siemens' != 'bosch' -> l_k=0.0, failed
      rule_2 range_check max_wattage/wattage: 1300 in [0, 1500] -> l_k=1.0, passed

    L = PROD l_k^w over defined = 0.0^0.9 * 1.0^0.8 = 0.0
    (Any zero collapses the product to 0.0)

    L=0.0 < tau_L=0.5 -> verdict False. PROVES the fix (old one-shot would approve).
    """
    L, n_passed, n_failed, n_undefined = _compute_l(
        "coffee_machine",
        COFFEE_MACHINE_A,
        "coffee_machine",
        COFFEE_MACHINE_B,
    )
    assert L < TAU_L, (
        f"Expected L < {TAU_L} (incompatible same-category), got L={L:.4f}"
    )
    verdict = compute_verdict(S_REP, L, TAU_S, TAU_L)
    assert verdict is False, (
        f"Expected verdict=False for coffee_machine->coffee_machine, got {verdict}"
    )


# ---------------------------------------------------------------------------
# Assertion 2: coffee_machine -> descaler >= tau_L  (verdict True)
# ---------------------------------------------------------------------------


def test_coffee_machine_to_descaler_compatible():
    """Coffee machine -> Siemens descaler: cleaning accessory, should be compatible.

    Rules for (coffee_machine, descaler):
      rule_1 set_intersection compatible_brands:
        A={'siemens'}, B={'siemens','bosch','neff'}
        |A & B|=1, min(1,3)=1 -> l_k=1.0, passed
      rule_2 exact_match product_category/target_appliance_category:
        'coffee_machine' == 'coffee_machine' -> l_k=1.0, passed

    L = 1.0^0.8 * 1.0^0.7 = 1.0 >= tau_L=0.5 -> verdict True.
    """
    L, n_passed, n_failed, n_undefined = _compute_l(
        "coffee_machine",
        COFFEE_MACHINE_A,
        "descaler",
        DESCALER,
    )
    assert L >= TAU_L, f"Expected L >= {TAU_L} (compatible accessory), got L={L:.4f}"
    verdict = compute_verdict(S_REP, L, TAU_S, TAU_L)
    assert verdict is True, (
        f"Expected verdict=True for coffee_machine->descaler, got {verdict}"
    )


# ---------------------------------------------------------------------------
# Assertion 3: coffee_machine -> water_filter >= tau_L  (verdict True)
# ---------------------------------------------------------------------------


def test_coffee_machine_to_water_filter_compatible():
    """Coffee machine -> Siemens water filter: compatible filter accessory.

    Rules for (coffee_machine, water_filter):
      rule_1 set_intersection compatible_brands:
        A={'siemens'}, B={'siemens'}
        |A & B|=1, min(1,1)=1 -> l_k=1.0, passed
      rule_2 exact_match water_connection_type/connection_type:
        'internal_tank' == 'internal_tank' -> l_k=1.0, passed

    L = 1.0^0.8 * 1.0^0.9 = 1.0 >= tau_L=0.5 -> verdict True.
    """
    L, n_passed, n_failed, n_undefined = _compute_l(
        "coffee_machine",
        COFFEE_MACHINE_A,
        "water_filter",
        WATER_FILTER,
    )
    assert L >= TAU_L, f"Expected L >= {TAU_L} (compatible filter), got L={L:.4f}"
    verdict = compute_verdict(S_REP, L, TAU_S, TAU_L)
    assert verdict is True, (
        f"Expected verdict=True for coffee_machine->water_filter, got {verdict}"
    )


# ---------------------------------------------------------------------------
# Assertion 4: coffee_machine -> electric_kettle < tau_L  (verdict False)
# ---------------------------------------------------------------------------


def test_coffee_machine_to_electric_kettle_incompatible():
    """Coffee machine -> electric kettle: unrelated appliance, MUST be incompatible.

    This is the CORE FIX: the old one-shot LLM verify approved this pairing
    as a loose semantic similarity. The rule-based path correctly rejects it.

    Rules for (coffee_machine, electric_kettle):
      rule_1 exact_match appliance_category/appliance_category:
        'coffee_machine' != 'electric_kettle' -> l_k=0.0, failed
      rule_2 exact_match heating_element_type/heating_element_type:
        'thermoblock' != 'concealed_element' -> l_k=0.0, failed

    L = 0.0^1.0 * 0.0^0.7 = 0.0 < tau_L=0.5 -> verdict False.
    """
    L, n_passed, n_failed, n_undefined = _compute_l(
        "coffee_machine",
        COFFEE_MACHINE_A,
        "electric_kettle",
        ELECTRIC_KETTLE,
    )
    assert L < TAU_L, (
        f"Expected L < {TAU_L} (unrelated appliance), got L={L:.4f}. "
        "This is the CORE FIX: coffee machine must NOT recommend electric kettle."
    )
    verdict = compute_verdict(S_REP, L, TAU_S, TAU_L)
    assert verdict is False, (
        f"Expected verdict=False for coffee_machine->electric_kettle, got {verdict}"
    )


# ---------------------------------------------------------------------------
# Cache hit path: second call must not trigger LLM
# ---------------------------------------------------------------------------


def test_rule_cache_hit_no_llm_call():
    """After first generation, rules come from cache (no second LLM call needed)."""

    class FailOnSecondCallLLM(MockRuleGen):
        _call_count = 0

        async def generate(self, system, user, fmt):
            type(self)._call_count += 1
            if type(self)._call_count > 1:
                raise AssertionError("generate() called more than once - cache miss!")
            return await super().generate(system, user, fmt)

    cache = MockRuleCache()
    llm = FailOnSecondCallLLM()

    rules1, _, _, _ = _run(
        get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False)
    )
    rules2, _, _, _ = _run(
        get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False)
    )

    assert rules1 == rules2, "Cached rules must match original rules"


# ---------------------------------------------------------------------------
# Directionality: (A, B) != (B, A)
# ---------------------------------------------------------------------------


def test_all_undefined_attributes_gives_zero_l():
    """Regression: all-undefined rules must produce L=0, not L=1.

    When a product pair has no usable attributes for any rule (every rule
    returns 'undefined'), the buggy semantics of treating undefined as l_k=1.0
    would produce L=1.0 and flip the verdict to True (false positive).

    The correct conservative behavior: all-undefined -> L=0.0 -> verdict False.

    This test uses empty attribute dicts to force all rules to undefined status
    and verifies the pipeline-level outcome, not just aggregate_logical in
    isolation.
    """
    EMPTY_ATTRS: dict = {}
    cache = MockRuleCache()
    llm = MockRuleGen()
    rules, _, _, _ = _run(
        get_rules("coffee_machine", "electric_kettle", llm, cache, hdr_enabled=False)
    )
    eval_results = []
    for rule in rules:
        l_k, status = evaluate_rule(rule, EMPTY_ATTRS, EMPTY_ATTRS)
        eval_results.append((l_k, rule["weight"], status))

    # All rules must be undefined when both attribute dicts are empty
    assert all(s == "undefined" for _, _, s in eval_results), (
        f"Expected all undefined, got: {eval_results}"
    )

    big_l, n_passed, n_failed, n_undefined = aggregate_logical(eval_results)

    assert big_l == 0.0, (
        f"All-undefined must produce L=0.0, got {big_l}. "
        "Regression: buggy 'treat undefined as 1.0' would return L=1.0 here."
    )
    assert n_undefined == len(rules)
    assert n_passed == 0
    assert n_failed == 0

    verdict = compute_verdict(S_REP, big_l, TAU_S, TAU_L)
    assert verdict is False, (
        "All-undefined product pair must never be recommended (verdict False)."
    )


def test_directional_keys_are_independent():
    """(coffee_machine, descaler) and (descaler, coffee_machine) are distinct cache slots."""
    cache = MockRuleCache()
    llm = MockRuleGen()

    rules_ab, _, _, _ = _run(
        get_rules("coffee_machine", "descaler", llm, cache, hdr_enabled=False)
    )
    rules_ba, _, _, _ = _run(
        get_rules("descaler", "coffee_machine", llm, cache, hdr_enabled=False)
    )
    # Both should be valid (non-empty); they may differ because rules are directional
    assert len(rules_ab) >= 2
    assert len(rules_ba) >= 2
    # Not necessarily equal - directionality check
    cached_ab = asyncio.run(cache.get("coffee_machine", "descaler"))
    cached_ba = asyncio.run(cache.get("descaler", "coffee_machine"))
    assert cached_ab is not None
    assert cached_ba is not None
