"""Retrieve and filter compatibility candidates from Typesense.

Verbatim logic from the old engine/candidates.py, with updated import path for
category_complements.json (now alongside this module).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os

import httpx

log = logging.getLogger(__name__)

_COMPLEMENT_MAP_PATH = os.path.join(
    os.path.dirname(__file__), "category_complements.json"
)
_complement_map: dict | None = None


def _load_complement_map() -> dict:
    """Lazy-load the category -> complementary-categories map (strategy 'cat_priors')."""
    global _complement_map
    if _complement_map is None:
        try:
            with open(_COMPLEMENT_MAP_PATH, encoding="utf-8") as f:
                _complement_map = json.load(f)
        except (OSError, json.JSONDecodeError):
            log.error("complement map missing at %s", _COMPLEMENT_MAP_PATH)
            _complement_map = {}
    return _complement_map


def _parse_attributes(doc: dict) -> dict:
    """Parse the Typesense attributes_json string field into a dict."""
    raw = doc.get("attributes_json")
    if not raw:
        return {}
    with contextlib.suppress(json.JSONDecodeError, TypeError):
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    return {}


def _normalize_doc(doc: dict) -> dict:
    """Normalize a Typesense product document into the engine's source/candidate shape."""
    doc["attributes"] = _parse_attributes(doc)
    doc.setdefault("compatibility_tags", [])
    doc.setdefault("categories", [])
    return doc


async def load_products(
    http_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    limit: int = 0,
    page_size: int = 250,
) -> list[dict]:
    """Load source products (with embeddings) from Typesense.

    Typesense is the source of truth: it holds the embedding vector, categories,
    parsed attributes and all fields the compatibility engine needs. The MySQL
    product_ai_data.normalized_json does NOT contain the embedding, so the
    engine reads products from Typesense.

    Args:
        http_client: Async HTTP client.
        base_url: Typesense base URL.
        api_key: Typesense API key.
        collection: Collection name.
        limit: Max products to load (0 = all).
        page_size: Documents per page.

    Returns:
        List of product dicts with embedding, attributes, categories, etc.
    """
    url = f"{base_url}/collections/{collection}/documents/search"
    headers = {"X-TYPESENSE-API-KEY": api_key}
    products: list[dict] = []
    page = 1

    while True:
        params = {
            "q": "*",
            "query_by": "name",
            "sort_by": "product_id:asc",
            "per_page": page_size,
            "page": page,
        }
        resp = await http_client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        if not hits:
            break
        for hit in hits:
            products.append(_normalize_doc(hit["document"]))
            if limit and len(products) >= limit:
                log.info(
                    "loaded %d products from typesense (limit reached)", len(products)
                )
                return products
        page += 1

    log.info("loaded %d products from typesense", len(products))
    return products


async def search_candidates(
    http_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    product_id: int,
    embedding: list[float],
    top_k: int = 50,
) -> list[dict]:
    """Search Typesense for top-K nearest neighbors by vector similarity.

    The source embedding is sent via POST multi_search (a 1024-dim vector does not
    fit in a GET query string). The returned vector_distance is the cosine
    distance; semantic similarity S(i,j) = cos = 1 - vector_distance is stored in
    _score (higher = more similar), matching article formula S(i,j)=cos(e_i,e_j).

    Returns:
        List of candidate documents with _score (semantic similarity) and parsed attributes.
    """
    url = f"{base_url}/multi_search"
    headers = {"X-TYPESENSE-API-KEY": api_key}
    vec_str = ",".join(str(v) for v in embedding)
    body = {
        "searches": [
            {
                "collection": collection,
                "q": "*",
                "vector_query": f"embedding:([{vec_str}], k:{top_k})",
                "exclude_fields": "embedding",
                "per_page": top_k,
            }
        ]
    }

    resp = await http_client.post(url, json=body, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    results_envelope = data.get("results", [{}])[0]

    results = []
    for hit in results_envelope.get("hits", []):
        doc = _normalize_doc(hit["document"])
        # cosine similarity = 1 - cosine distance
        doc["_score"] = 1.0 - float(hit.get("vector_distance", 1.0))
        results.append(doc)

    log.info("product=%d candidates=%d", product_id, len(results))
    return results


async def search_candidates_cat_priors(
    http_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    source: dict,
    embedding: list[float],
    top_k: int = 50,
) -> list[dict]:
    """Retrieval strategy A: restrict vector search to the source product's
    COMPLEMENTARY categories (from the category->complement map), so accessories/
    companions are surfaced instead of similar products. Ranked by semantic similarity
    within those categories. Falls back to empty if the source has no mapped complements.
    """
    cmap = _load_complement_map()
    comps: set[str] = set()
    for cat in source.get("categories") or []:
        comps.update(cmap.get(cat, []))
    product_id = source.get("product_id")
    if not comps:
        log.info("product=%s cat_priors candidates=0 (no complements)", product_id)
        return []

    vec_str = ",".join(str(v) for v in embedding)
    filter_by = "categories:=[" + ",".join("`" + c + "`" for c in sorted(comps)) + "]"
    body = {
        "searches": [
            {
                "collection": collection,
                "q": "*",
                "vector_query": f"embedding:([{vec_str}], k:{top_k})",
                "filter_by": filter_by,
                "exclude_fields": "embedding",
                "per_page": top_k,
            }
        ]
    }
    resp = await http_client.post(
        f"{base_url}/multi_search", json=body, headers={"X-TYPESENSE-API-KEY": api_key}
    )
    resp.raise_for_status()
    envelope = resp.json().get("results", [{}])[0]
    results = []
    for hit in envelope.get("hits", []):
        doc = _normalize_doc(hit["document"])
        doc["_score"] = 1.0 - float(hit.get("vector_distance", 1.0))
        results.append(doc)
    log.info(
        "product=%s cat_priors candidates=%d (complements=%d)",
        product_id,
        len(results),
        len(comps),
    )
    return results


async def retrieve_candidates(
    strategy: str,
    http_client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    source: dict,
    top_k: int,
) -> list[dict]:
    """Dispatch to the configured retrieval strategy (RETRIEVAL_STRATEGY)."""
    embedding = source.get("embedding", [])
    if not embedding:
        return []
    if strategy == "cat_priors":
        return await search_candidates_cat_priors(
            http_client, base_url, api_key, collection, source, embedding, top_k
        )
    return await search_candidates(
        http_client,
        base_url,
        api_key,
        collection,
        source["product_id"],
        embedding,
        top_k,
    )


def filter_candidates(
    source: dict, candidates: list[dict], drop_same_category: bool = True
) -> list[dict]:
    """Filter out self-references and (optionally) same-category products.

    The category fields carry the full path (leaf + parent/top names), so a phone
    and its case share the broad parent ('Mobile Phones & Gadgets') even though
    their leaf categories differ. With drop_same_category=True any shared category
    -- including that parent -- removes the candidate, which wrongly discards real
    accessories. For cat_priors the complement map already guarantees the candidate
    sits in a complementary leaf category, so the caller passes
    drop_same_category=False and only self is removed.

    Args:
        source: Source product dict with product_id and categories.
        candidates: List of candidate dicts.
        drop_same_category: When True, drop candidates sharing any category with the
            source (used by the similarity strategy to avoid recommending near-
            duplicates). When False, only the self-reference is removed.

    Returns:
        Filtered list of candidates.
    """
    source_cats = set(source.get("categories", []))
    source_id = source.get("product_id")

    return [
        c
        for c in candidates
        if c.get("product_id") != source_id
        and not (
            drop_same_category and set(c.get("categories", [])) & source_cats
        )
    ]
