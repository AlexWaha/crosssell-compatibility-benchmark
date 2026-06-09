"""Pure, deterministic rule evaluation for JIT-ontology compatibility scoring.

No I/O. Every function is side-effect-free and safe to call from tests without
any database or network dependencies.

Article formula: L(i,j,c) = PROD_{k in DEFINED} l_k^{w_k}
where DEFINED = rules whose evaluation status is 'passed' or 'failed'
(not 'undefined' due to missing attributes).
"""

from __future__ import annotations

import math
import re
import unicodedata


# ---------------------------------------------------------------------------
# Value normalisation helpers
# ---------------------------------------------------------------------------

_UNIT_PATTERN = re.compile(
    r"\s*(kg|g|lb|lbs|oz|cm|mm|m|km|in|inch|inches|l|ml|w|kw|mw|v|mv|kv|"
    r"a|ma|ka|hz|khz|mhz|ghz|rpm|db|dbm|"
    r"watts?|volts?|amps?|amperes?|ohms?|farads?|henrys?)\s*$",
    re.IGNORECASE,
)


def normalize_value(value: object) -> str:
    """Normalize a scalar attribute value for comparison.

    Steps:
    1. Convert to str, strip whitespace.
    2. Unicode NFKC normalization.
    3. Lowercase.
    4. Strip trailing unit suffixes (e.g. ' W', 'kg', 'V').
    5. Strip residual whitespace.

    Args:
        value: Any scalar (str, int, float, bool, None).

    Returns:
        Normalised string ready for comparison.
    """
    if value is None:
        return ""
    raw = unicodedata.normalize("NFKC", str(value)).strip().lower()
    raw = _UNIT_PATTERN.sub("", raw).strip()
    return raw


def _to_float(value: object) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _to_set(value: object) -> set[str] | None:
    """Convert a value to a set of normalized strings.

    Accepts lists, comma-separated strings, or single values.
    Returns None if the result is empty (missing/unusable).
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        items = {normalize_value(v) for v in value if v is not None}
    else:
        items = {normalize_value(s) for s in str(value).split(",") if s.strip()}
    return items if items else None


# ---------------------------------------------------------------------------
# Individual rule evaluators
# ---------------------------------------------------------------------------

_STATUS_PASSED = "passed"
_STATUS_FAILED = "failed"
_STATUS_UNDEFINED = "undefined"


def _eval_exact_match(rule: dict, attrs_i: dict, attrs_j: dict) -> tuple[float, str]:
    """exact_match: l_k = 1.0 if normalised values are equal, else 0.0.

    Returns 'undefined' when either attribute is absent or empty after
    normalisation (missing data cannot produce a verdict).

    Args:
        rule: Rule dict with keys 'attribute_a', 'attribute_b'.
        attrs_i: Attribute dict of source product i.
        attrs_j: Attribute dict of target product j.

    Returns:
        Tuple of (l_k float, status str).
    """
    a = normalize_value(attrs_i.get(rule["attribute_a"]))
    b = normalize_value(attrs_j.get(rule["attribute_b"]))
    if not a or not b:
        return 0.0, _STATUS_UNDEFINED
    if a == b:
        return 1.0, _STATUS_PASSED
    return 0.0, _STATUS_FAILED


def _eval_range_check(rule: dict, attrs_i: dict, attrs_j: dict) -> tuple[float, str]:
    """range_check: l_k = 1.0 if attrs_j[attribute_b] is within [min, max]
    derived from attrs_i[attribute_a], else 0.0.

    The source attribute is parsed as either a single max value (numeric) or
    a 'min-max' formatted string. The target attribute must be numeric.

    Args:
        rule: Rule dict with 'attribute_a' (source range/max) and
            'attribute_b' (target value).
        attrs_i: Attribute dict of source product i.
        attrs_j: Attribute dict of target product j.

    Returns:
        Tuple of (l_k float, status str).
    """
    raw_a = attrs_i.get(rule["attribute_a"])
    raw_b = attrs_j.get(rule["attribute_b"])
    if raw_a is None or raw_b is None:
        return 0.0, _STATUS_UNDEFINED

    target = _to_float(normalize_value(raw_b))
    if target is None:
        return 0.0, _STATUS_UNDEFINED

    # Parse source: try 'min-max' or 'min to max' then single numeric (treated as max, min=0)
    src_str = normalize_value(raw_a)
    range_match = re.match(r"^(-?[\d.]+)\s*(?:-|\s*to\s*)\s*(-?[\d.]+)$", src_str)
    if range_match:
        lo = _to_float(range_match.group(1))
        hi = _to_float(range_match.group(2))
    else:
        lo = 0.0
        hi = _to_float(src_str)

    if lo is None or hi is None:
        return 0.0, _STATUS_UNDEFINED
    if lo <= target <= hi:
        return 1.0, _STATUS_PASSED
    return 0.0, _STATUS_FAILED


def _eval_set_intersection(
    rule: dict, attrs_i: dict, attrs_j: dict
) -> tuple[float, str]:
    """set_intersection: l_k = |A intersect B| / min(|A|, |B|).

    Returns 'undefined' when either side is empty/missing.

    Args:
        rule: Rule dict with 'attribute_a' and 'attribute_b'.
        attrs_i: Attribute dict of source product i.
        attrs_j: Attribute dict of target product j.

    Returns:
        Tuple of (l_k float, status str).
    """
    a = _to_set(attrs_i.get(rule["attribute_a"]))
    b = _to_set(attrs_j.get(rule["attribute_b"]))
    if a is None or b is None:
        return 0.0, _STATUS_UNDEFINED
    intersection = len(a & b)
    denom = min(len(a), len(b))
    if denom == 0:
        return 0.0, _STATUS_UNDEFINED
    score = intersection / denom
    status = _STATUS_PASSED if score > 0.0 else _STATUS_FAILED
    return score, status


def _eval_regex_match(rule: dict, attrs_i: dict, attrs_j: dict) -> tuple[float, str]:
    """regex_match: l_k = 1.0 if attrs_j[attribute_b] matches the pattern
    stored in attrs_i[attribute_a], else 0.0.

    Returns 'undefined' when either attribute is missing.

    Args:
        rule: Rule dict with 'attribute_a' (pattern source) and
            'attribute_b' (value to match against pattern).
        attrs_i: Attribute dict of source product i.
        attrs_j: Attribute dict of target product j.

    Returns:
        Tuple of (l_k float, status str).
    """
    pattern_raw = attrs_i.get(rule["attribute_a"])
    value_raw = attrs_j.get(rule["attribute_b"])
    if pattern_raw is None or value_raw is None:
        return 0.0, _STATUS_UNDEFINED
    try:
        matched = bool(re.search(str(pattern_raw), str(value_raw), re.IGNORECASE))
    except re.error:
        return 0.0, _STATUS_UNDEFINED
    if matched:
        return 1.0, _STATUS_PASSED
    return 0.0, _STATUS_FAILED


_EVALUATORS = {
    "exact_match": _eval_exact_match,
    "range_check": _eval_range_check,
    "set_intersection": _eval_set_intersection,
    "regex_match": _eval_regex_match,
}


def evaluate_rule(rule: dict, attrs_i: dict, attrs_j: dict) -> tuple[float, str]:
    """Evaluate a single compatibility rule against a product pair.

    Dispatches to the appropriate type-specific evaluator.

    Args:
        rule: Rule dict with at minimum keys 'type', 'attribute_a', 'attribute_b'.
        attrs_i: Normalised attribute dict of source product i.
        attrs_j: Normalised attribute dict of target product j.

    Returns:
        (l_k, status) where l_k is in [0.0, 1.0] and
        status is 'passed' | 'failed' | 'undefined'.

    Raises:
        ValueError: If rule['type'] is not one of the four known types.
    """
    rule_type = rule.get("type", "")
    evaluator = _EVALUATORS.get(rule_type)
    if evaluator is None:
        raise ValueError(
            f"Unknown rule type '{rule_type}'. Allowed: {list(_EVALUATORS)}"
        )
    return evaluator(rule, attrs_i, attrs_j)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_logical(
    results: list[tuple[float, float, str]],
    l_agg: str = "weighted_product",
) -> tuple[float, int, int, int]:
    """Aggregate per-rule evaluations into a single logical score L.

    Article formula (weighted_product):
        L = PROD_{k in DEFINED} l_k ^ w_k
    where DEFINED excludes rules whose status is 'undefined'.

    If ALL rules are undefined -> L = 0.0 (conservative; fixes the loose
    one-shot approval problem).

    Args:
        results: List of (l_k, weight, status) tuples for each rule.
        l_agg: Aggregation mode:
            - 'weighted_product' (default): L = PROD l_k^w over defined rules.
            - 'product': L = PROD l_k over defined rules (unweighted).

    Returns:
        Tuple of (L, n_passed, n_failed, n_undefined).
    """
    n_passed = sum(1 for _, _, s in results if s == _STATUS_PASSED)
    n_failed = sum(1 for _, _, s in results if s == _STATUS_FAILED)
    n_undefined = sum(1 for _, _, s in results if s == _STATUS_UNDEFINED)

    defined = [(score, w, s) for score, w, s in results if s != _STATUS_UNDEFINED]
    if not defined:
        # All undefined -> conservative zero (fixes loose one-shot approval)
        return 0.0, n_passed, n_failed, n_undefined

    log_sum = 0.0
    for score, weight, _ in defined:
        exponent = weight if l_agg == "weighted_product" else 1.0
        if score <= 0.0:
            # Any zero in the product collapses L to 0
            return 0.0, n_passed, n_failed, n_undefined
        log_sum += exponent * math.log(score)

    big_l = max(0.0, min(1.0, math.exp(log_sum)))
    return big_l, n_passed, n_failed, n_undefined


# ---------------------------------------------------------------------------
# Context code
# ---------------------------------------------------------------------------

_TYPE_CONTEXT_MAP: dict[str, str] = {
    "cable": "cable",
    "mount": "mount",
    "accessory": "accessory",
    "charger": "charger",
    "adapter": "adapter",
    "case": "case",
    "filter": "filter",
    "cleaning": "cleaning",
    "descaler": "cleaning",
    "water_filter": "filter",
    "kettle": "appliance",
    "electric_kettle": "appliance",
    "coffee_machine": "appliance",
    "espresso_machine": "appliance",
    "coffee_grinder": "appliance",
}


def compute_context_code(rules: list[dict], type_a: str, type_b: str) -> str:
    """Derive a deterministic context code from the type pair and rules.

    Looks up type_a and type_b in a fixed vocabulary. If neither maps to a
    known context, falls back to 'accessory'.

    Args:
        rules: List of rule dicts (not used for mapping, reserved for future
            rule-driven context inference).
        type_a: Product type of source product.
        type_b: Product type of target product.

    Returns:
        Context code string (e.g. 'cable', 'accessory', 'filter').
    """
    ta = type_a.lower().replace(" ", "_")
    tb = type_b.lower().replace(" ", "_")
    ctx_a = _TYPE_CONTEXT_MAP.get(ta)
    ctx_b = _TYPE_CONTEXT_MAP.get(tb)
    # Prefer the non-appliance context code if one exists
    for ctx in (ctx_a, ctx_b):
        if ctx and ctx != "appliance":
            return ctx
    if ctx_a:
        return ctx_a
    if ctx_b:
        return ctx_b
    return "accessory"
