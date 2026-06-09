"""Product normalizer: UPS / eta operator implementation.

Provides RealNormalizer (calls LLM via RealLLM.generate), MockNormalizer (deterministic,
$0, suitable for CI and smoke testing), get_normalizer factory, and the top-level
normalize_product() function that wraps normalization with metrics collection.

The source_hash (SHA-256 over canonical serialization of all inputs including prompt and
model version) enables idempotent/resumable processing: if the hash matches the stored
value in product_ai_data, the product is skipped without any LLM call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

from app.services.llm.verifier import LLMError, RealLLM
from app.services.normalization.prompt import (
    PROMPT_VERSION,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    build_user_prompt,
    parse_normalization,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source hash
# ---------------------------------------------------------------------------


def compute_source_hash(
    product: dict,
    raw_attrs: dict[str, str],
    categories: list[str],
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """Compute a deterministic SHA-256 hash of all normalization inputs.

    The hash covers: product fields (id, name, description, brand), raw attributes,
    categories, model name, and prompt version. A change to any input - including
    a prompt or model version bump - produces a different hash and therefore
    invalidates the cached result in product_ai_data.

    Args:
        product: Product row dict. Only stable fields (id, name, description, brand)
            are included; mutable fields like product_type are excluded.
        raw_attrs: Dict of attribute_name -> attribute_value from product_attributes.
        categories: List of category names (sorted for stability).
        model: LLM model identifier string.
        prompt_version: Prompt version string (e.g. 'norm_v1').

    Returns:
        Hex-encoded SHA-256 digest (64 characters).
    """
    # Defensive: raw_attrs MUST be a dict {attribute_name: attribute_value}.
    # If a list of {name, value} dicts is passed (schema mismatch), convert it so
    # compute_source_hash does not crash. The authoritative contract is dict - this
    # guard catches regressions introduced by callers, not the primary data path.
    if isinstance(raw_attrs, list):
        raw_attrs = {
            item["name"]: item.get("value", "")
            for item in raw_attrs
            if isinstance(item, dict) and "name" in item
        }

    canonical = {
        "product_id": product.get("product_id"),
        "name": product.get("name") or "",
        "description": (product.get("description") or "")[:800],
        "brand": product.get("brand") or "",
        "raw_attrs": dict(sorted(raw_attrs.items())),
        "categories": sorted(categories),
        "model": model,
        "prompt_version": prompt_version,
    }
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Real normalizer
# ---------------------------------------------------------------------------


class RealNormalizer:
    """Calls the LLM (via RealLLM.generate) to produce a UPS for each product.

    Args:
        llm: A RealLLM instance (cfg + async OpenAI client).
        model: Model identifier string (from cfg.PRIMARY_MODEL).
    """

    def __init__(self, llm: RealLLM, model: str) -> None:
        self._llm = llm
        self._model = model

    async def normalize(
        self,
        product: dict,
        raw_attrs: dict[str, str],
        categories: list[str],
    ) -> tuple[dict | None, int, int]:
        """Normalize one product via the LLM.

        Args:
            product: Product row dict.
            raw_attrs: Raw attribute dict.
            categories: Category names list.

        Returns:
            Tuple of (parsed_ups_dict_or_None, tokens_in, tokens_out).

        Raises:
            LLMError: On provider error (caller handles retry logic).
        """
        # Defensive: ensure raw_attrs is a dict before prompt construction.
        if isinstance(raw_attrs, list):
            raw_attrs = {
                item["name"]: item.get("value", "")
                for item in raw_attrs
                if isinstance(item, dict) and "name" in item
            }

        user_prompt = build_user_prompt(product, raw_attrs, categories)
        raw, ti, to = await self._llm.generate(
            SYSTEM_PROMPT, user_prompt, RESPONSE_FORMAT
        )
        parsed = parse_normalization(raw)
        if parsed is None:
            log.warning(
                "normalization parse failure product_id=%s", product.get("product_id")
            )
        return parsed, ti, to


# ---------------------------------------------------------------------------
# Mock normalizer
# ---------------------------------------------------------------------------

# Simple keyword heuristics for deterministic product_type derivation in mock mode.
_TYPE_HEURISTICS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"coffee|espresso|cappuccino|kaffee", re.I), "coffee_machine"),
    (re.compile(r"laptop|notebook", re.I), "laptop"),
    (re.compile(r"smartphone|iphone|galaxy|phone", re.I), "smartphone"),
    (re.compile(r"tablet|ipad", re.I), "tablet"),
    (re.compile(r"headphone|earphone|earbuds|kopfhoerer", re.I), "headphone"),
    (re.compile(r"television|tv|fernseher", re.I), "television"),
    (re.compile(r"printer|drucker", re.I), "printer"),
    (re.compile(r"camera|kamera", re.I), "camera"),
    (re.compile(r"keyboard|tastatur", re.I), "keyboard"),
    (re.compile(r"mouse|maus", re.I), "computer_mouse"),
    (re.compile(r"cable|kabel", re.I), "cable"),
    (re.compile(r"charger|ladeger", re.I), "charger"),
    (re.compile(r"case|hulle|cover", re.I), "case"),
    (re.compile(r"kettle|wasserkocher", re.I), "electric_kettle"),
    (re.compile(r"vacuum|staubsauger", re.I), "vacuum_cleaner"),
    (re.compile(r"refrigerator|fridge|kuhlschrank", re.I), "refrigerator"),
    (re.compile(r"washing|waschmaschine", re.I), "washing_machine"),
    (re.compile(r"monitor|display|bildschirm", re.I), "monitor"),
    (re.compile(r"router|wlan", re.I), "router"),
    (re.compile(r"capsule|kapsel|pod", re.I), "coffee_capsule"),
]


def _derive_product_type(product: dict, categories: list[str]) -> str:
    """Derive a plausible product_type from name and categories using heuristics.

    Args:
        product: Product row dict.
        categories: Category names list.

    Returns:
        Lowercase snake_case product type string.
    """
    text = " ".join(
        [
            product.get("name") or "",
            product.get("brand") or "",
        ]
        + categories
    )
    for pattern, ptype in _TYPE_HEURISTICS:
        if pattern.search(text):
            return ptype
    return "product"


def _derive_tags(product: dict, product_type: str, categories: list[str]) -> list[str]:
    """Derive compatibility tags from product data.

    Args:
        product: Product row dict.
        product_type: Already-derived product type string.
        categories: Category names list.

    Returns:
        List of lowercase compatibility tag strings.
    """
    tags: list[str] = [product_type]
    brand = (product.get("brand") or "").lower().strip()
    if brand:
        tags.append(brand)
    # Add category slugs as tags (simplified: lowercase, replace spaces)
    for cat in categories[:3]:
        slug = cat.lower().replace(" ", "_").replace("-", "_")
        if slug and slug not in tags:
            tags.append(slug)
    return tags


class MockNormalizer:
    """Deterministic $0 normalizer for CI and smoke testing.

    Derives product_type and tags from the raw product name and categories using
    keyword heuristics so the mock output is plausible. Returns 0 tokens (no LLM call).

    Args:
        model: Model identifier string (used for source_hash; not actually called).
    """

    def __init__(self, model: str = "mock") -> None:
        self._model = model

    async def normalize(
        self,
        product: dict,
        raw_attrs: dict[str, str],
        categories: list[str],
    ) -> tuple[dict | None, int, int]:
        """Produce a deterministic valid UPS without any LLM call.

        Args:
            product: Product row dict.
            raw_attrs: Raw attribute dict.
            categories: Category names list.

        Returns:
            Tuple of (ups_dict, 0, 0) - tokens are always zero for mock.
        """
        # Defensive: ensure raw_attrs is a dict.
        if isinstance(raw_attrs, list):
            raw_attrs = {
                item["name"]: item.get("value", "")
                for item in raw_attrs
                if isinstance(item, dict) and "name" in item
            }

        product_type = _derive_product_type(product, categories)
        tags = _derive_tags(product, product_type, categories)

        # Build a minimal normalized_json from raw_attrs (up to 10 attrs)
        normalized: dict[str, str] = {}
        for k, v in list(raw_attrs.items())[:10]:
            norm_key = re.sub(r"\s+", "_", k.lower().strip())
            normalized[norm_key] = str(v).strip()

        name = (product.get("name") or "").strip()
        brand = (product.get("brand") or "").strip()
        tag_str = " ".join(tags[:5])
        embedding_text = f"{name} {product_type} {brand} {tag_str}".strip()
        # Truncate to 300 chars
        embedding_text = embedding_text[:300]

        ups = {
            "product_type": product_type,
            "normalized_json": normalized,
            "compatibility_tags": tags,
            "embedding_text": embedding_text,
        }
        return ups, 0, 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_normalizer(cfg, client=None) -> RealNormalizer | MockNormalizer:
    """Create a normalizer based on LLM_MODE config.

    Args:
        cfg: Settings instance.
        client: Async OpenAI client (required for real mode, ignored for mock).

    Returns:
        MockNormalizer when cfg.LLM_MODE == 'mock', RealNormalizer otherwise.
    """
    if cfg.LLM_MODE == "mock":
        return MockNormalizer(model="mock")
    from app.services.llm.verifier import RealLLM

    llm = RealLLM(cfg, client)
    return RealNormalizer(llm=llm, model=cfg.PRIMARY_MODEL)


# ---------------------------------------------------------------------------
# Top-level normalize_product
# ---------------------------------------------------------------------------


async def normalize_product(
    normalizer: RealNormalizer | MockNormalizer,
    product: dict,
    raw_attrs: dict[str, str],
    categories: list[str],
    model: str,
    prompt_version: str = PROMPT_VERSION,
) -> dict:
    """Normalize one product and collect metrics.

    Computes source_hash, calls normalizer.normalize(), and returns a result dict
    containing the UPS fields plus a 'metrics' sub-dict for instrumentation.

    Args:
        normalizer: RealNormalizer or MockNormalizer instance.
        product: Product row dict (product_id, name, description, brand).
        raw_attrs: Raw attribute dict from product_attributes table.
        categories: Category names the product belongs to.
        model: LLM model identifier (used in source_hash).
        prompt_version: Prompt version string (used in source_hash).

    Returns:
        Dict with keys:
            source_hash (str): SHA-256 hex digest of inputs.
            product_type (str | None): Derived product type.
            normalized_json (dict | None): Canonical attribute dict.
            compatibility_tags (list | None): Compatibility facet tags.
            embedding_text (str | None): Compact retrieval string.
            success (bool): True if parse produced a valid UPS.
            metrics (dict): Instrumentation with keys:
                attrs_before (int): Raw attribute count before normalization.
                attrs_after (int): Normalized attribute count after normalization.
                attrs_missing_before (int): Raw attrs with empty/None values.
                filled (int): Attrs present in normalized_json but absent/empty in raw.
                fill_rate (float): filled / max(attrs_before, 1).
                llm_calls (int): Number of LLM calls made (0 for mock/failure).
                tokens_input (int): Input tokens consumed.
                tokens_output (int): Output tokens consumed.
    """
    # Defensive: ensure raw_attrs is a dict before any attribute-level access.
    # compute_source_hash also guards, but normalize_product accesses raw_attrs
    # directly (values(), keys()) so the guard must live here too.
    if isinstance(raw_attrs, list):
        raw_attrs = {
            item["name"]: item.get("value", "")
            for item in raw_attrs
            if isinstance(item, dict) and "name" in item
        }

    source_hash = compute_source_hash(
        product, raw_attrs, categories, model, prompt_version
    )

    attrs_before = len(raw_attrs)
    attrs_missing_before = sum(1 for v in raw_attrs.values() if not v)
    raw_keys = set(re.sub(r"\s+", "_", k.lower().strip()) for k in raw_attrs.keys())

    llm_calls = 0
    tokens_input = 0
    tokens_output = 0
    ups = None

    try:
        ups, ti, to = await normalizer.normalize(product, raw_attrs, categories)
        llm_calls = 1
        tokens_input = ti
        tokens_output = to
    except LLMError as exc:
        log.error(
            "normalize_product LLMError product_id=%s: %s",
            product.get("product_id"),
            exc,
        )

    if ups is None:
        return {
            "source_hash": source_hash,
            "product_type": None,
            "normalized_json": None,
            "compatibility_tags": None,
            "embedding_text": None,
            "success": False,
            "metrics": {
                "attrs_before": attrs_before,
                "attrs_after": 0,
                "attrs_missing_before": attrs_missing_before,
                "filled": 0,
                "fill_rate": 0.0,
                "llm_calls": llm_calls,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
            },
        }

    normalized_keys = set(ups["normalized_json"].keys())
    attrs_after = len(normalized_keys)

    # Canonical value set from raw attributes, used for value-side matching below.
    # Values are lowercased and stripped so minor casing differences do not inflate filled.
    raw_values: set[str] = {str(v).lower().strip() for v in raw_attrs.values() if v}

    # JITOE "filled" definition (article-grade metric):
    #
    #   An attribute in the normalized output is counted as FILLED (i.e. genuinely
    #   inferred by the LLM, not merely relabelled) if and only if BOTH conditions hold:
    #
    #     (a) its canonicalized key does NOT match any canonicalized raw key, AND
    #     (b) its value does NOT match any raw value (after normalize_value).
    #
    #   Condition (a) alone over-counts when the LLM renames/canonicalizes an attribute
    #   (e.g. German "Leistung" -> canonical "power_w").  Such a rename is a key
    #   relabelling, not new knowledge.  Condition (b) catches that case: if the
    #   normalized value equals an existing raw value the attribute is a rename, not an
    #   inference.  Only when a key is genuinely new AND its value has no raw counterpart
    #   do we treat it as a JITOE-filled attribute.
    #
    #   fill_rate = filled / max(attrs_before, 1), capped at 1.0.
    #   The cap prevents a product with zero raw attributes from reporting fill_rate > 1.
    #   The raw 'filled' count is always preserved in metrics for transparency.
    filled = sum(
        1
        for k, v in ups["normalized_json"].items()
        if k not in raw_keys and str(v).lower().strip() not in raw_values
    )
    fill_rate = min(filled / max(attrs_before, 1), 1.0)

    return {
        "source_hash": source_hash,
        "product_type": ups["product_type"],
        "normalized_json": ups["normalized_json"],
        "compatibility_tags": ups["compatibility_tags"],
        "embedding_text": ups["embedding_text"],
        "success": True,
        "metrics": {
            "attrs_before": attrs_before,
            "attrs_after": attrs_after,
            "attrs_missing_before": attrs_missing_before,
            "filled": filled,
            "fill_rate": fill_rate,
            "llm_calls": llm_calls,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        },
    }
