"""JIT-ontology rule generation.

Generates compatibility rules for (type_a, type_b) product pairs on demand,
caches them (Redis + table), and applies the HDR / evidence gate to drop
hallucinated rules whose attributes never appear in the product vocabulary.

Rule-gen prompt ported verbatim from project/docs/prompts/jit-ontology.md.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.compatibility.cache import RuleCache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt constants (verbatim from jit-ontology.md)
# ---------------------------------------------------------------------------

RULEGEN_SYSTEM_PROMPT = (
    "You are a technical compatibility expert. Given two product types, generate rules "
    "that determine whether products of these types are technically compatible."
)

# Use json_object (not json_schema strict) because gpt-5-nano burns the entire
# max_completion_tokens budget on reasoning when strict json_schema is used,
# leaving zero tokens for actual output (finish_reason=length, content='').
# _validate_rules() enforces structural correctness after parsing.
RULEGEN_RESPONSE_FORMAT = {"type": "json_object"}

_VALID_RULE_TYPES = {"exact_match", "range_check", "set_intersection", "regex_match"}
_MIN_RULES = 2
_MAX_RULES = 8


def build_rulegen_prompt(
    type_a: str,
    type_b: str,
    vocab_a: list[str] | None = None,
    vocab_b: list[str] | None = None,
) -> str:
    """Build the user prompt for rule generation.

    When vocab_a / vocab_b are provided (per-type attribute key lists from
    product_ai_data), they are injected into the prompt so the LLM generates
    rules that reference REAL attribute names rather than invented ones.

    Args:
        type_a: Source product type label.
        type_b: Target product type label.
        vocab_a: Ordered list of real attribute keys for type_a products
            (most frequent first). None means no vocab constraint in the prompt.
        vocab_b: Ordered list of real attribute keys for type_b products
            (most frequent first). None means no vocab constraint in the prompt.

    Returns:
        Formatted user prompt string.
    """
    lines = [
        f"Source product type: {type_a}",
        f"Target product type: {type_b}",
        "",
    ]

    if vocab_a is not None or vocab_b is not None:
        lines.append("AVAILABLE ATTRIBUTES (use ONLY these names):")
        if vocab_a is not None:
            lines.append(f"  {type_a} attributes: {', '.join(vocab_a)}")
        if vocab_b is not None:
            lines.append(f"  {type_b} attributes: {', '.join(vocab_b)}")
        lines.append("")
        lines.append(
            "CRITICAL: attribute_a MUST be one of the listed "
            f"{type_a} attributes. "
            "attribute_b MUST be one of the listed "
            f"{type_b} attributes. "
            "Do NOT invent attribute names. "
            "If no meaningful compatibility rule can be built from the available "
            "attributes, return an empty rules array."
        )
        lines.append(
            "Prefer attributes likely to be shared or comparable across both types "
            "(e.g. brand, compatible_brands, connector_type, voltage, capacity). "
            "Brand-based rules (exact_match on brand, or set_intersection on "
            "compatible_brands) are reliable when both types carry that attribute."
        )
        lines.append("")

    lines += [
        "Return a JSON object with a 'rules' array of 2-8 rules. Each rule has:",
        '- "id": unique string like "rule_1"',
        '- "type": one of exact_match | range_check | set_intersection | regex_match',
        '- "attribute_a": attribute name on source product (lowercase snake_case)',
        '- "attribute_b": attribute name on target product (lowercase snake_case)',
        '- "weight": 0.0 to 1.0 (1.0 = critical incompatibility, 0.1 = minor)',
        '- "description": human-readable explanation',
        "",
        "Focus on TECHNICAL compatibility, not commercial fit.",
        "Weight guidance:",
        "  1.0 = critical: violation means guaranteed incompatibility",
        "  0.7-0.9 = important: likely incompatible if violated",
        "  0.3-0.6 = moderate: reduced functionality if violated",
        "  0.1-0.2 = minor: cosmetic or convenience issue",
        "",
        "Prefer exact_match for connectors/interfaces/sockets.",
        "Prefer range_check for power/capacity/dimensions.",
        "Prefer set_intersection for protocols/standards/features.",
        "Order rules by weight descending (most critical first).",
    ]

    return "\n".join(lines)


def _validate_rules(raw_rules: list) -> list[dict]:
    """Validate and sanitize a list of raw rule dicts from LLM output.

    - Checks required fields are present.
    - Validates type is in the four allowed types.
    - Clamps weight to [0.0, 1.0].
    - Enforces count in [_MIN_RULES, _MAX_RULES].

    Malformed individual rules (missing required keys, unknown type, non-numeric
    weight) are DROPPED with a warning rather than crashing the whole type-pair.
    A ValueError is raised only after the loop if the number of surviving valid
    rules falls below _MIN_RULES.

    Args:
        raw_rules: List of dicts as returned by the LLM.

    Returns:
        Validated and sanitized list of rule dicts.

    Raises:
        ValueError: If raw_rules is not a list, or if surviving valid rule
            count is below _MIN_RULES.
    """
    if not isinstance(raw_rules, list):
        raise ValueError(f"rules must be a list, got {type(raw_rules).__name__}")
    if len(raw_rules) > _MAX_RULES:
        log.warning(
            "rule-gen returned %d rules (max %d), truncating",
            len(raw_rules),
            _MAX_RULES,
        )
        raw_rules = raw_rules[:_MAX_RULES]

    validated: list[dict] = []
    required_keys = {
        "id",
        "type",
        "attribute_a",
        "attribute_b",
        "weight",
        "description",
    }
    for idx, rule in enumerate(raw_rules):
        if not isinstance(rule, dict):
            log.warning("rule[%d] is not a dict, skipping", idx)
            continue
        missing = required_keys - rule.keys()
        if missing:
            log.warning("rule[%d] missing keys %s, skipping", idx, missing)
            continue
        rule_type = rule["type"]
        if rule_type not in _VALID_RULE_TYPES:
            log.warning(
                "rule[%d] has unknown type '%s' (allowed: %s), skipping",
                idx,
                rule_type,
                _VALID_RULE_TYPES,
            )
            continue
        # Clamp weight
        try:
            weight = float(rule["weight"])
        except (TypeError, ValueError):
            log.warning("rule[%d] weight is not numeric, skipping", idx)
            continue
        rule = dict(rule)
        rule["weight"] = max(0.0, min(1.0, weight))
        validated.append(rule)

    if len(validated) < _MIN_RULES:
        raise ValueError(
            f"rule-gen produced only {len(validated)} valid rules "
            f"(after dropping malformed), minimum is {_MIN_RULES}"
        )

    return validated


def _apply_hdr_gate(
    rules: list[dict],
    vocab_a: set[str] | None,
    vocab_b: set[str] | None,
) -> tuple[list[dict], int, int]:
    """Apply HDR / evidence gate: drop rules whose attributes are not in the vocab.

    'Vocab' is the set of attribute keys that actually occur in products of
    that type. Rules whose attribute_a is not in vocab_a (or attribute_b not
    in vocab_b) are considered hallucinated and excluded.

    If either vocab is None, that side's gate is skipped (no vocab available).

    Approximation note (article section 4):
        The paper defines the HDR gate using a similarity criterion
        q(t) >= tau_q, where q(t) is the semantic similarity between a
        generated attribute token t and the nearest token in the product
        vocabulary. TAU_Q is defined in config but is NOT yet wired here.
        The current implementation uses binary attribute presence/absence
        as a cheaper approximation of that criterion: an attribute either
        occurs in the vocab or it does not (no partial-match scoring).
        TODO: wire full q(t) similarity when embedding-level vocab lookup
        is available (see article section 4 for the full formulation).

    Args:
        rules: Validated rule list.
        vocab_a: Set of attribute keys present for type_a products. None = skip.
        vocab_b: Set of attribute keys present for type_b products. None = skip.

    Returns:
        Tuple of (filtered_rules, evidence_claims, hallucinated_claims).
    """
    filtered: list[dict] = []
    hallucinated = 0
    for rule in rules:
        attr_a_ok = vocab_a is None or rule["attribute_a"] in vocab_a
        attr_b_ok = vocab_b is None or rule["attribute_b"] in vocab_b
        if attr_a_ok and attr_b_ok:
            filtered.append(rule)
        else:
            hallucinated += 1
            log.debug(
                "HDR gate dropped rule id=%s attr_a=%s (in_vocab=%s) "
                "attr_b=%s (in_vocab=%s)",
                rule.get("id"),
                rule["attribute_a"],
                attr_a_ok,
                rule["attribute_b"],
                attr_b_ok,
            )

    evidence_claims = len(filtered)
    return filtered, evidence_claims, hallucinated


def _source_hash(type_a: str, type_b: str, prompt: str) -> str:
    """Compute a deterministic SHA-256 hash of the rule-gen inputs."""
    content = f"{type_a}|{type_b}|{prompt}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def get_rules(
    type_a: str,
    type_b: str,
    llm,
    cache: "RuleCache",
    hdr_enabled: bool = True,
    vocab_a: set[str] | None = None,
    vocab_b: set[str] | None = None,
) -> tuple[list[dict], int, int, bool]:
    """Retrieve or generate compatibility rules for a (type_a, type_b) pair.

    Cache hit path: return cached rules immediately (no LLM call, $0).
    Cache miss path: call llm.generate(), validate, apply HDR gate, cache.

    Directionality: (type_a, type_b) and (type_b, type_a) are treated as
    distinct cache keys because rules may be asymmetric.

    Args:
        type_a: Source product type.
        type_b: Target product type.
        llm: LLM client with a generate(system, user, response_format) method.
        cache: RuleCache implementation (MockRuleCache, TableRuleCache, etc.).
        hdr_enabled: Whether to apply the HDR / evidence gate.
        vocab_a: Optional set of attribute keys for type_a products.
        vocab_b: Optional set of attribute keys for type_b products.

    Returns:
        Tuple of (rules, evidence_claims, hallucinated_claims, was_cache_miss).
        was_cache_miss is True only when a real LLM generate() call was made.
        Use this flag to increment llm_calls only on actual LLM invocations
        (not on cache hits, which cost $0 and do not consume tokens).

    Raises:
        ValueError: If LLM output fails validation.
        LLMError: If the LLM call fails (propagated from llm.generate).
    """
    cached = await cache.get(type_a, type_b)
    if cached is not None:
        log.debug("rule cache hit type_a=%s type_b=%s", type_a, type_b)
        # HDR gate still applied on cache hits so vocab changes take effect
        if hdr_enabled and (vocab_a is not None or vocab_b is not None):
            filtered, ev, hall = _apply_hdr_gate(cached, vocab_a, vocab_b)
            return filtered, ev, hall, False
        return cached, len(cached), 0, False

    user_prompt = build_rulegen_prompt(
        type_a,
        type_b,
        vocab_a=list(vocab_a) if vocab_a is not None else None,
        vocab_b=list(vocab_b) if vocab_b is not None else None,
    )
    s_hash = _source_hash(type_a, type_b, user_prompt)

    log.info("rule cache miss, calling LLM type_a=%s type_b=%s", type_a, type_b)
    raw_data, _ti, _to = await llm.generate(
        RULEGEN_SYSTEM_PROMPT, user_prompt, RULEGEN_RESPONSE_FORMAT
    )

    # raw_data may be a dict (from JSON schema) or a pre-parsed list
    if isinstance(raw_data, dict):
        raw_rules = raw_data.get("rules", [])
    elif isinstance(raw_data, list):
        raw_rules = raw_data
    else:
        raw_rules = []

    rules = _validate_rules(raw_rules)

    evidence_claims = len(rules)
    hallucinated_claims = 0

    if hdr_enabled:
        rules, evidence_claims, hallucinated_claims = _apply_hdr_gate(
            rules, vocab_a, vocab_b
        )
        if len(rules) < _MIN_RULES:
            log.warning(
                "HDR gate reduced rules to %d (< %d) for %s->%s; "
                "using all pre-gate rules",
                len(rules),
                _MIN_RULES,
                type_a,
                type_b,
            )
            # Restore the full validated rule list for evaluation so the
            # pipeline does not crash with too few rules. Preserve the
            # evidence_claims / hallucinated_claims that were computed by the
            # ORIGINAL gate run (with real vocabs) - re-running _apply_hdr_gate
            # with (None, None) would reset hallucinated_claims to 0, corrupting
            # the HDR audit metric used by the paper.
            rules = _validate_rules(raw_rules)

    generated_by = getattr(llm, "_model_id", "llm")
    await cache.set(
        type_a, type_b, rules, generated_by=generated_by, source_hash=s_hash
    )

    log.info(
        "generated %d rules for %s->%s evidence=%d hallucinated=%d",
        len(rules),
        type_a,
        type_b,
        evidence_claims,
        hallucinated_claims,
    )
    return rules, evidence_claims, hallucinated_claims, True


# ---------------------------------------------------------------------------
# Mock rule sets for $0 testing (hand-authored, deterministic)
# ---------------------------------------------------------------------------

# Hand-authored rule sets for the sanity test fixture type-pairs.
# These mirror what a real LLM would generate for these pairs and are used
# exclusively by MockLLM.generate() in mock/test scenarios.

_MOCK_RULE_SETS: dict[tuple[str, str], list[dict]] = {
    # coffee_machine -> coffee_machine: same-type appliances
    # Incompatible by design: a coffee machine is not a cross-sell FOR another
    # coffee machine. Rules target brand, wattage range, and voltage - all
    # likely to diverge across two distinct machines.
    ("coffee_machine", "coffee_machine"): [
        {
            "id": "rule_1",
            "type": "exact_match",
            "attribute_a": "brand",
            "attribute_b": "brand",
            "weight": 0.9,
            "description": "Same-category products need identical brand for cross-sell",
        },
        {
            "id": "rule_2",
            "type": "range_check",
            "attribute_a": "max_wattage",
            "attribute_b": "wattage",
            "weight": 0.8,
            "description": "Target wattage must fit within source appliance range",
        },
    ],
    # coffee_machine -> descaler: accessory/cleaning supply - compatible
    ("coffee_machine", "descaler"): [
        {
            "id": "rule_1",
            "type": "set_intersection",
            "attribute_a": "compatible_brands",
            "attribute_b": "compatible_brands",
            "weight": 0.8,
            "description": "Descaler must list the coffee machine brand as compatible",
        },
        {
            "id": "rule_2",
            "type": "exact_match",
            "attribute_a": "product_category",
            "attribute_b": "target_appliance_category",
            "weight": 0.7,
            "description": "Descaler must target the coffee machine appliance category",
        },
    ],
    # coffee_machine -> water_filter: accessory - compatible
    ("coffee_machine", "water_filter"): [
        {
            "id": "rule_1",
            "type": "set_intersection",
            "attribute_a": "compatible_brands",
            "attribute_b": "compatible_brands",
            "weight": 0.8,
            "description": "Water filter must list coffee machine brand as compatible",
        },
        {
            "id": "rule_2",
            "type": "exact_match",
            "attribute_a": "water_connection_type",
            "attribute_b": "connection_type",
            "weight": 0.9,
            "description": "Filter connection type must match machine water inlet",
        },
    ],
    # coffee_machine -> electric_kettle: different appliance - incompatible
    ("coffee_machine", "electric_kettle"): [
        {
            "id": "rule_1",
            "type": "exact_match",
            "attribute_a": "appliance_category",
            "attribute_b": "appliance_category",
            "weight": 1.0,
            "description": "Appliance category must match for cross-sell compatibility",
        },
        {
            "id": "rule_2",
            "type": "exact_match",
            "attribute_a": "heating_element_type",
            "attribute_b": "heating_element_type",
            "weight": 0.7,
            "description": "Heating element type must be compatible for accessories",
        },
    ],
}


class MockRuleGen:
    """Returns deterministic hand-authored rules for test fixture type-pairs.

    Used as the LLM backend in $0 / mock tests. Implements the same
    generate(system, user, response_format) interface as RealLLM.

    Falls back to a generic 2-rule set for unknown type pairs.

    Attributes:
        _model_id: Identifier surfaced by get_rules for the cache audit trail.
    """

    _model_id = "mock_rule_gen"

    async def generate(
        self, system_prompt: str, user_prompt: str, response_format: dict
    ) -> tuple[dict, int, int]:
        """Return mock rule-gen output for the type pair extracted from user_prompt.

        Args:
            system_prompt: Ignored in mock mode.
            user_prompt: Contains 'Source product type: X' and
                'Target product type: Y' lines.
            response_format: Ignored in mock mode.

        Returns:
            Tuple of ({"rules": [...]}, tokens_in, tokens_out).
        """
        type_a, type_b = self._parse_types(user_prompt)
        rules = _MOCK_RULE_SETS.get((type_a, type_b)) or self._fallback_rules(
            type_a, type_b
        )
        return {"rules": rules}, 50, 150

    def _parse_types(self, user_prompt: str) -> tuple[str, str]:
        """Extract type_a and type_b from the formatted user prompt."""
        type_a = ""
        type_b = ""
        for line in user_prompt.splitlines():
            if line.startswith("Source product type:"):
                type_a = line.split(":", 1)[1].strip()
            elif line.startswith("Target product type:"):
                type_b = line.split(":", 1)[1].strip()
        return type_a, type_b

    def _fallback_rules(self, type_a: str, type_b: str) -> list[dict]:
        """Generic 2-rule fallback for unknown type pairs."""
        return [
            {
                "id": "rule_1",
                "type": "exact_match",
                "attribute_a": "brand",
                "attribute_b": "brand",
                "weight": 0.5,
                "description": f"Brand compatibility between {type_a} and {type_b}",
            },
            {
                "id": "rule_2",
                "type": "set_intersection",
                "attribute_a": "compatible_with",
                "attribute_b": "compatible_with",
                "weight": 0.5,
                "description": f"Shared compatibility tags for {type_a} and {type_b}",
            },
        ]

    # Keep verify() compatible interface so MockRuleGen can double as a full
    # mock LLM in tests that only exercise rule-gen (not product verification)
    async def verify(
        self, source: dict, candidates: list[dict]
    ) -> tuple[list[dict], int, int]:
        """Stub verify - not used in JIT-ontology path."""
        return [], 0, 0
