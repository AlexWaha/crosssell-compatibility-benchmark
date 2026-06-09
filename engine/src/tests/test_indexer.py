"""Unit tests for app.services.indexing (schema, indexer, MockEmbedder).

All tests run at $0 / mock - no LLM, no real embeddings, no DB, no Typesense.
Covers:
- MockEmbedder determinism and dimension
- build_document produces the exact candidates.py doc shape
- attributes_json is valid JSON parseable as a dict
- schema dict has embedding float[] num_dim=1024 cosine + all required fields
- candidates._parse_attributes: valid JSON, malformed JSON, empty, non-dict
- candidates._normalize_doc: attributes key set, compatibility_tags/categories defaults
- candidates.filter_candidates: self exclusion, same-category exclusion, cross-category pass

Run: docker exec avtc_engine python -m pytest tests/test_indexer.py -v
"""

from __future__ import annotations

import asyncio
import json

from app.services.indexing.indexer import MockEmbedder, _unit_vector, build_document
from app.services.indexing.schema import PRODUCTS_SCHEMA_FIELDS, build_collection_schema
from app.services.retrieval.candidates import (
    _normalize_doc,
    _parse_attributes,
    filter_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


# Minimal product_ai_data row (as returned by the DB join query)
_AI_ROW = {
    "product_id": 42,
    "product_type": "coffee_machine",
    "normalized_json": {"max_power_w": "1500", "pressure_bar": "15"},
    "compatibility_tags": ["coffee_machine", "siemens", "espresso"],
    "embedding_text": "Siemens EQ.6 Plus coffee machine 1500W espresso",
}

# Minimal products table row
_PRODUCT_ROW = {
    "product_id": 42,
    "slug": "siemens-eq6-plus",
    "name": "Siemens EQ.6 Plus Coffee Machine",
    "description": "Fully automatic espresso machine.",
    "brand": "Siemens",
    "price": 799.99,
    "image_path": "catalog/product/siemens-eq6.jpg",
    "status": 1,
}

_CATEGORIES = ["Coffee Machines", "Espresso"]

_EMBEDDING_DIM = 1024


# ---------------------------------------------------------------------------
# _unit_vector
# ---------------------------------------------------------------------------


def test_unit_vector_dim():
    """_unit_vector must produce a vector of exactly dim elements."""
    vec = _unit_vector("test text", dim=_EMBEDDING_DIM)
    assert len(vec) == _EMBEDDING_DIM


def test_unit_vector_unit_norm():
    """_unit_vector must produce a unit-norm vector (L2 norm ~1.0)."""
    import math

    vec = _unit_vector("test text", dim=_EMBEDDING_DIM)
    norm = math.sqrt(sum(v * v for v in vec))
    assert abs(norm - 1.0) < 1e-6, f"L2 norm={norm}, expected 1.0"


def test_unit_vector_deterministic():
    """_unit_vector must return the same vector for the same input."""
    v1 = _unit_vector("same text", dim=_EMBEDDING_DIM)
    v2 = _unit_vector("same text", dim=_EMBEDDING_DIM)
    assert v1 == v2


def test_unit_vector_differs_on_different_text():
    """_unit_vector must return different vectors for different inputs."""
    v1 = _unit_vector("text A", dim=_EMBEDDING_DIM)
    v2 = _unit_vector("text B", dim=_EMBEDDING_DIM)
    assert v1 != v2


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


def test_mock_embedder_dim():
    """MockEmbedder.embed_batch must return vectors of exactly 1024 dimensions."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    vecs = _run(emb.embed_batch(["product A", "product B"]))
    assert len(vecs) == 2
    assert all(len(v) == _EMBEDDING_DIM for v in vecs)


def test_mock_embedder_deterministic():
    """MockEmbedder must return identical vectors for identical inputs across calls."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    texts = ["product A", "product B"]
    vecs1 = _run(emb.embed_batch(texts))
    vecs2 = _run(emb.embed_batch(texts))
    assert vecs1 == vecs2, "MockEmbedder is not deterministic"


def test_mock_embedder_different_texts_different_vectors():
    """MockEmbedder must produce distinct vectors for distinct texts."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    vecs = _run(emb.embed_batch(["text one", "text two"]))
    assert vecs[0] != vecs[1], "distinct texts must produce distinct vectors"


def test_mock_embedder_single_text():
    """MockEmbedder must handle a single-element batch."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    vecs = _run(emb.embed_batch(["single"]))
    assert len(vecs) == 1
    assert len(vecs[0]) == _EMBEDDING_DIM


def test_mock_embedder_empty_text_returns_vector():
    """MockEmbedder must return a vector even for empty text (no crash)."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    vecs = _run(emb.embed_batch([""]))
    assert len(vecs) == 1
    assert len(vecs[0]) == _EMBEDDING_DIM


# ---------------------------------------------------------------------------
# build_document - shape contract (candidates.py)
# ---------------------------------------------------------------------------

# These are the keys that candidates._normalize_doc and search logic expect.
_REQUIRED_KEYS = {
    "id",
    "product_id",
    "name",
    "brand",
    "product_type",
    "description",
    "compatibility_tags",
    "categories",
    "price",
    "embedding",
    "attributes_json",
    "embedding_text",
    "image",
}


def _make_doc(ai_row=None, product_row=None, categories=None, embedding=None):
    """Helper: build a document with sensible defaults."""
    emb = MockEmbedder(dim=_EMBEDDING_DIM)
    vec = asyncio.run(emb.embed_batch(["test"]))[0]
    return _run(
        build_document(
            ai_row=ai_row or _AI_ROW,
            product_row=product_row or _PRODUCT_ROW,
            categories=categories if categories is not None else _CATEGORIES,
            embedding=embedding or vec,
        )
    )


def test_build_document_has_all_required_keys():
    """build_document must produce a dict with all candidates.py-required keys."""
    doc = _make_doc()
    missing = _REQUIRED_KEYS - set(doc.keys())
    assert not missing, f"Missing document keys: {missing}"


def test_build_document_id_is_string():
    """Document id must be a str (Typesense requires string primary key)."""
    doc = _make_doc()
    assert isinstance(doc["id"], str)
    assert doc["id"] == str(_PRODUCT_ROW["product_id"])


def test_build_document_product_id_is_int():
    """product_id must be an int (Typesense int32 field)."""
    doc = _make_doc()
    assert isinstance(doc["product_id"], int)
    assert doc["product_id"] == _PRODUCT_ROW["product_id"]


def test_build_document_embedding_len_1024():
    """embedding must be a list of exactly 1024 floats."""
    doc = _make_doc()
    assert isinstance(doc["embedding"], list)
    assert len(doc["embedding"]) == _EMBEDDING_DIM


def test_build_document_attributes_json_is_valid_json_dict():
    """attributes_json must be a JSON string parseable as a dict."""
    doc = _make_doc()
    raw = doc["attributes_json"]
    assert isinstance(raw, str), "attributes_json must be a string"
    parsed = json.loads(raw)
    assert isinstance(parsed, dict), "attributes_json must parse to a dict"


def test_build_document_attributes_json_contains_normalized_data():
    """attributes_json must contain the keys from ai_row normalized_json."""
    doc = _make_doc()
    parsed = json.loads(doc["attributes_json"])
    assert "max_power_w" in parsed
    assert parsed["max_power_w"] == "1500"


def test_build_document_categories_is_list():
    """categories must be a list of strings (string[] in Typesense)."""
    doc = _make_doc()
    assert isinstance(doc["categories"], list)
    assert doc["categories"] == _CATEGORIES


def test_build_document_compatibility_tags_is_list():
    """compatibility_tags must be a list of strings."""
    doc = _make_doc()
    assert isinstance(doc["compatibility_tags"], list)
    assert doc["compatibility_tags"] == _AI_ROW["compatibility_tags"]


def test_build_document_price_is_float():
    """price must be a float."""
    doc = _make_doc()
    assert isinstance(doc["price"], float)


def test_build_document_empty_categories():
    """build_document must handle a product with no categories."""
    doc = _make_doc(categories=[])
    assert doc["categories"] == []


def test_build_document_empty_compatibility_tags():
    """build_document must handle ai_row with null/empty compatibility_tags."""
    ai_no_tags = {**_AI_ROW, "compatibility_tags": None}
    doc = _make_doc(ai_row=ai_no_tags)
    assert doc["compatibility_tags"] == []


def test_build_document_null_normalized_json():
    """build_document must handle ai_row with null normalized_json gracefully."""
    ai_null_json = {**_AI_ROW, "normalized_json": None}
    doc = _make_doc(ai_row=ai_null_json)
    parsed = json.loads(doc["attributes_json"])
    assert parsed == {}


def test_build_document_json_string_normalized_json():
    """build_document must handle normalized_json already serialized as a JSON string."""
    ai_str_json = {**_AI_ROW, "normalized_json": '{"key": "val"}'}
    doc = _make_doc(ai_row=ai_str_json)
    parsed = json.loads(doc["attributes_json"])
    assert parsed == {"key": "val"}


def test_build_document_embedding_text_propagated():
    """embedding_text from ai_row must appear in the document."""
    doc = _make_doc()
    assert doc["embedding_text"] == _AI_ROW["embedding_text"]


def test_build_document_name_from_product_row():
    """name must come from product_row, not ai_row."""
    doc = _make_doc()
    assert doc["name"] == _PRODUCT_ROW["name"]


# ---------------------------------------------------------------------------
# Schema - PRODUCTS_SCHEMA_FIELDS contract
# ---------------------------------------------------------------------------


def _field(name: str) -> dict | None:
    """Return the schema field dict for the given field name, or None."""
    for f in PRODUCTS_SCHEMA_FIELDS:
        if f["name"] == name:
            return f
    return None


def test_schema_has_embedding_field():
    """Schema must have an 'embedding' field of type float[]."""
    emb = _field("embedding")
    assert emb is not None, "embedding field missing from schema"
    assert emb["type"] == "float[]"


def test_schema_embedding_num_dim_1024():
    """Schema embedding must have num_dim=1024."""
    emb = _field("embedding")
    assert emb is not None
    assert emb.get("num_dim") == 1024, (
        f"expected num_dim=1024, got {emb.get('num_dim')}"
    )


def test_schema_embedding_cosine():
    """Schema embedding must use cosine distance."""
    emb = _field("embedding")
    assert emb is not None
    assert emb.get("vec_dist") == "cosine", (
        f"expected cosine, got {emb.get('vec_dist')}"
    )


def test_schema_has_categories_string_array():
    """Schema must have categories as string[]."""
    cats = _field("categories")
    assert cats is not None, "categories field missing from schema"
    assert cats["type"] == "string[]"


def test_schema_has_compatibility_tags_string_array():
    """Schema must have compatibility_tags as string[]."""
    tags = _field("compatibility_tags")
    assert tags is not None, "compatibility_tags field missing from schema"
    assert tags["type"] == "string[]"


def test_schema_has_attributes_json_string():
    """Schema must have attributes_json as string."""
    f = _field("attributes_json")
    assert f is not None, "attributes_json field missing from schema"
    assert f["type"] == "string"


def test_schema_has_product_id_int32():
    """Schema must have product_id as int32."""
    f = _field("product_id")
    assert f is not None, "product_id field missing from schema"
    assert f["type"] == "int32"


def test_schema_has_price_float():
    """Schema must have price as float."""
    f = _field("price")
    assert f is not None, "price field missing from schema"
    assert f["type"] == "float"


def test_schema_has_all_candidates_fields():
    """Schema must contain all fields that candidates.py reads from documents."""
    # Fields directly accessed by candidates._normalize_doc / search functions
    required = {
        "product_id",
        "name",
        "brand",
        "product_type",
        "description",
        "compatibility_tags",
        "categories",
        "price",
        "embedding",
        "attributes_json",
        "embedding_text",
        "image",
    }
    schema_names = {f["name"] for f in PRODUCTS_SCHEMA_FIELDS}
    missing = required - schema_names
    assert not missing, f"Schema missing fields required by candidates.py: {missing}"


def test_build_collection_schema_name():
    """build_collection_schema must use the provided name."""
    schema = build_collection_schema("products_v2")
    assert schema["name"] == "products_v2"


def test_build_collection_schema_has_fields():
    """build_collection_schema must include all PRODUCTS_SCHEMA_FIELDS."""
    schema = build_collection_schema("products_v2")
    assert schema["fields"] is PRODUCTS_SCHEMA_FIELDS
    assert len(schema["fields"]) > 0


# ---------------------------------------------------------------------------
# candidates._parse_attributes - contract tests
# ---------------------------------------------------------------------------


def test_parse_attributes_valid_json_dict():
    """_parse_attributes must return a dict for a valid JSON object string."""
    doc = {"attributes_json": '{"voltage": "220V", "power_w": "1500"}'}
    result = _parse_attributes(doc)
    assert isinstance(result, dict)
    assert result["voltage"] == "220V"
    assert result["power_w"] == "1500"


def test_parse_attributes_empty_string_returns_empty_dict():
    """_parse_attributes must return {} for an empty attributes_json string."""
    doc = {"attributes_json": ""}
    result = _parse_attributes(doc)
    assert result == {}


def test_parse_attributes_missing_key_returns_empty_dict():
    """_parse_attributes must return {} when attributes_json key is absent."""
    result = _parse_attributes({})
    assert result == {}


def test_parse_attributes_none_value_returns_empty_dict():
    """_parse_attributes must return {} when attributes_json is None."""
    doc = {"attributes_json": None}
    result = _parse_attributes(doc)
    assert result == {}


def test_parse_attributes_malformed_json_returns_empty_dict():
    """_parse_attributes must return {} (not raise) on malformed JSON."""
    doc = {"attributes_json": "{not valid json"}
    result = _parse_attributes(doc)
    assert result == {}


def test_parse_attributes_json_array_returns_empty_dict():
    """_parse_attributes must return {} when the JSON root is a list, not a dict."""
    doc = {"attributes_json": '["a", "b"]'}
    result = _parse_attributes(doc)
    assert result == {}


def test_parse_attributes_empty_json_object_returns_empty_dict():
    """_parse_attributes must return {} for an empty JSON object '{}'."""
    doc = {"attributes_json": "{}"}
    result = _parse_attributes(doc)
    assert result == {}


# ---------------------------------------------------------------------------
# candidates._normalize_doc - contract tests
# ---------------------------------------------------------------------------


def _make_raw_doc(**overrides) -> dict:
    """Return a minimal raw Typesense document for _normalize_doc tests."""
    base = {
        "product_id": 1,
        "name": "Widget",
        "brand": "ACME",
        "product_type": "gadget",
        "description": "A test widget.",
        "compatibility_tags": ["gadget", "acme"],
        "categories": ["Electronics"],
        "price": 9.99,
        "attributes_json": '{"color": "red"}',
        "embedding_text": "Widget ACME gadget",
        "image": "img/widget.jpg",
    }
    base.update(overrides)
    return base


def test_normalize_doc_sets_attributes_key():
    """_normalize_doc must add an 'attributes' key parsed from attributes_json."""
    doc = _normalize_doc(_make_raw_doc())
    assert "attributes" in doc
    assert isinstance(doc["attributes"], dict)
    assert doc["attributes"]["color"] == "red"


def test_normalize_doc_attributes_is_dict_from_valid_json():
    """_normalize_doc attributes must contain the keys from attributes_json."""
    doc = _normalize_doc(_make_raw_doc(attributes_json='{"key": "val"}'))
    assert doc["attributes"] == {"key": "val"}


def test_normalize_doc_compatibility_tags_default_to_empty_list():
    """_normalize_doc must ensure compatibility_tags defaults to [] when absent."""
    raw = _make_raw_doc()
    raw.pop("compatibility_tags")
    doc = _normalize_doc(raw)
    assert doc["compatibility_tags"] == []


def test_normalize_doc_categories_default_to_empty_list():
    """_normalize_doc must ensure categories defaults to [] when absent."""
    raw = _make_raw_doc()
    raw.pop("categories")
    doc = _normalize_doc(raw)
    assert doc["categories"] == []


def test_normalize_doc_preserves_existing_compatibility_tags():
    """_normalize_doc must not overwrite an already-present compatibility_tags list."""
    doc = _normalize_doc(_make_raw_doc(compatibility_tags=["tag1", "tag2"]))
    assert doc["compatibility_tags"] == ["tag1", "tag2"]


def test_normalize_doc_preserves_existing_categories():
    """_normalize_doc must not overwrite an already-present categories list."""
    doc = _normalize_doc(_make_raw_doc(categories=["Cat A", "Cat B"]))
    assert doc["categories"] == ["Cat A", "Cat B"]


def test_normalize_doc_attributes_malformed_json_gives_empty_dict():
    """_normalize_doc must set attributes={} when attributes_json is malformed."""
    doc = _normalize_doc(_make_raw_doc(attributes_json="{bad"))
    assert doc["attributes"] == {}


# ---------------------------------------------------------------------------
# candidates.filter_candidates - contract tests
# ---------------------------------------------------------------------------


def _make_candidate(product_id: int, categories: list[str]) -> dict:
    """Return a minimal candidate doc for filter_candidates tests."""
    return {
        "product_id": product_id,
        "categories": categories,
        "name": f"p{product_id}",
    }


def test_filter_candidates_removes_self():
    """filter_candidates must exclude the source product from its own candidates."""
    source = {"product_id": 1, "categories": ["Electronics"]}
    candidates = [
        _make_candidate(1, ["Accessories"]),
        _make_candidate(2, ["Accessories"]),
    ]
    result = filter_candidates(source, candidates)
    ids = [c["product_id"] for c in result]
    assert 1 not in ids
    assert 2 in ids


def test_filter_candidates_removes_same_category():
    """filter_candidates must exclude candidates that share a category with source."""
    source = {"product_id": 10, "categories": ["Coffee Machines"]}
    candidates = [
        _make_candidate(11, ["Coffee Machines"]),  # same category - excluded
        _make_candidate(12, ["Accessories"]),  # different - kept
        _make_candidate(13, ["Coffee Machines", "Accessories"]),  # overlap - excluded
    ]
    result = filter_candidates(source, candidates)
    ids = [c["product_id"] for c in result]
    assert 11 not in ids
    assert 13 not in ids
    assert 12 in ids


def test_filter_candidates_keeps_cross_category_candidates():
    """filter_candidates must keep candidates with no shared categories."""
    source = {"product_id": 5, "categories": ["Laptops"]}
    candidates = [
        _make_candidate(6, ["Mice"]),
        _make_candidate(7, ["Keyboards"]),
        _make_candidate(8, ["Monitors"]),
    ]
    result = filter_candidates(source, candidates)
    assert len(result) == 3


def test_filter_candidates_empty_input_returns_empty():
    """filter_candidates must return [] when given an empty candidate list."""
    source = {"product_id": 1, "categories": ["A"]}
    result = filter_candidates(source, [])
    assert result == []


def test_filter_candidates_all_same_category_returns_empty():
    """filter_candidates must return [] when all candidates share the source's category."""
    source = {"product_id": 1, "categories": ["Printers"]}
    candidates = [
        _make_candidate(2, ["Printers"]),
        _make_candidate(3, ["Printers"]),
    ]
    result = filter_candidates(source, candidates)
    assert result == []


def test_filter_candidates_score_preserved():
    """filter_candidates must not strip _score from kept candidates."""
    source = {"product_id": 1, "categories": ["Laptops"]}
    c = _make_candidate(2, ["Mice"])
    c["_score"] = 0.75
    result = filter_candidates(source, [c])
    assert len(result) == 1
    assert result[0]["_score"] == 0.75
