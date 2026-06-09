"""Normalization prompt definitions for the UPS (Universal Product Specification) operator.

Defines the system prompt, structured response format (strict json_schema), user prompt
builder, and tolerant response parser for the LLM-based product normalization pipeline.

The normalizer implements the eta() operator from the article: given raw product data,
produce a canonical UPS with normalized attributes (SI units, lowercase snake_case keys,
enums from controlled vocab, set-valued attrs as arrays) and RAG-fill missing attributes
ONLY when evidenced in the provided text. Unknown attributes are omitted, never hallucinated.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

PROMPT_VERSION = "norm_v1"

SYSTEM_PROMPT = (
    "You are a product data normalization expert implementing the UPS (Universal Product"
    " Specification) operator.\n\n"
    "Given a product's raw name, description, and attributes, output a normalized product"
    " specification in strict JSON.\n\n"
    "Rules:\n"
    "1. product_type: lowercase snake_case singular noun (e.g. coffee_machine,"
    " smartphone, laptop, usb_cable). Use the most specific type that is clearly"
    " evidenced by the product data.\n"
    "2. normalized_json: dict of normalized attribute key-value pairs.\n"
    "   - Keys: lowercase snake_case (e.g. max_power_w, screen_size_inch,"
    " connection_type).\n"
    "   - Numeric values: separate the number from the unit. Store the number as a"
    " string (e.g. '1500', '13.3') and encode the unit in the key name (e.g."
    " max_power_w, screen_diagonal_inch, weight_kg).\n"
    "   - Enum values: lowercase string from a controlled vocabulary (e.g. connection"
    " type: 'usb_a', 'usb_c', 'hdmi', 'bluetooth', 'wifi'; color: 'black', 'white',"
    " 'silver'; os: 'android', 'ios', 'windows', 'macos', 'linux').\n"
    "   - Set-valued attributes (e.g. supported_brands, compatible_models,"
    " capsule_types): JSON array of lowercase strings.\n"
    "3. compatibility_tags: array of lowercase strings identifying compatibility-relevant"
    " facets for cross-selling retrieval (e.g. ['espresso', 'nespresso_compatible',"
    " 'milk_frother'] for a coffee machine).\n"
    "4. embedding_text: compact retrieval string (under 300 chars) combining product"
    " name, type, key attributes, and compatibility tags. Used for semantic vector"
    " search. Example: 'coffee machine espresso 15 bar 1500w nespresso compatible"
    " milk frother'.\n"
    "5. RAG-fill: if an attribute is NOT present in the raw data but is strongly"
    " evidenced by the product name or description, you MAY infer it. ONLY infer"
    " what is clearly evidenced. If uncertain, OMIT the attribute entirely. Never"
    " hallucinate values that cannot be grounded in the provided text.\n\n"
    "Output ONLY valid JSON, no markdown fences. Follow the exact schema provided."
)

_UPS_ATTRS_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": {"type": "string"},
}

_UPS_TAGS_SCHEMA: dict = {
    "type": "array",
    "items": {"type": "string"},
}

RESPONSE_FORMAT: dict = {
    # Use json_object (not json_schema) because normalized_json has dynamic attribute keys
    # chosen by the LLM. json_schema strict mode requires every property to be explicitly
    # declared, which is incompatible with an open-ended attribute dict. json_schema with
    # strict=False caused gpt-5-nano to echo the schema structure instead of filling it in.
    # The system prompt and parse_normalization() enforce structural correctness instead.
    "type": "json_object",
}


def build_user_prompt(
    product: dict,
    raw_attrs: dict[str, str],
    categories: list[str],
) -> str:
    """Build the user-facing normalization prompt for one product.

    Args:
        product: Product row dict with keys: product_id, name, description, brand,
            product_type (may be empty).
        raw_attrs: Dict of attribute_name -> attribute_value from product_attributes.
        categories: List of category names the product belongs to.

    Returns:
        Formatted user prompt string.
    """
    lines: list[str] = []

    name = product.get("name") or "Unknown"
    lines.append(f"Product name: {name}")

    if product.get("brand"):
        lines.append(f"Brand: {product['brand']}")

    if categories:
        lines.append(f"Categories: {', '.join(categories)}")

    if product.get("description"):
        # Truncate long descriptions to avoid token bloat
        desc = (product["description"] or "").strip()
        if len(desc) > 800:
            desc = desc[:800] + "..."
        if desc:
            lines.append(f"Description: {desc}")

    if raw_attrs:
        attrs_str = json.dumps(raw_attrs, ensure_ascii=False)
        lines.append(f"Raw attributes: {attrs_str}")

    lines.append(
        "\nNormalize this product into the UPS schema. "
        "RAG-fill missing but evidenced attributes. "
        "Omit unknowns. Never hallucinate."
    )

    return "\n".join(lines)


def parse_normalization(raw: dict | str | None) -> dict | None:
    """Parse and validate a normalization response from the LLM.

    Tolerant: accepts a raw dict (from generate()) or a JSON string.
    Returns None on any structural failure so the caller can handle gracefully.

    Args:
        raw: LLM response - either a parsed dict or raw JSON string.

    Returns:
        Validated dict with keys product_type, normalized_json,
        compatibility_tags, embedding_text, or None on failure.
    """
    if raw is None:
        return None

    if isinstance(raw, str):
        try:
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(text)
        except (json.JSONDecodeError, IndexError):
            log.error("parse_normalization: failed to parse JSON: %.200s", raw)
            return None
    elif isinstance(raw, dict):
        data = raw
    else:
        log.error("parse_normalization: unexpected type %s", type(raw))
        return None

    required = {
        "product_type",
        "normalized_json",
        "compatibility_tags",
        "embedding_text",
    }
    missing = required - data.keys()
    if missing:
        log.error("parse_normalization: missing required fields: %s", missing)
        return None

    if not isinstance(data.get("product_type"), str) or not data["product_type"]:
        log.error("parse_normalization: product_type must be non-empty string")
        return None

    if not isinstance(data.get("normalized_json"), dict):
        log.error("parse_normalization: normalized_json must be a dict")
        return None

    if not isinstance(data.get("compatibility_tags"), list):
        log.error("parse_normalization: compatibility_tags must be a list")
        return None

    if not isinstance(data.get("embedding_text"), str):
        log.error("parse_normalization: embedding_text must be a string")
        return None

    return {
        "product_type": str(data["product_type"]).strip().lower().replace(" ", "_"),
        "normalized_json": data["normalized_json"],
        "compatibility_tags": [
            str(t).strip().lower() for t in data["compatibility_tags"]
        ],
        "embedding_text": str(data["embedding_text"]).strip(),
    }
