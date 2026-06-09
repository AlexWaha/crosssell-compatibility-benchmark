"""Typesense collection schema for products_v2.

Defines the exact field set that candidates.py expects, with a 1024-dim cosine
embedding vector. Provides an idempotent ensure_collection helper that creates
or recreates the collection via the Typesense REST API.

The schema reproduces the live `products` collection field layout so
candidates.py, search_candidates, and search_candidates_cat_priors all work
unchanged against products_v2.
"""

from __future__ import annotations

import json
import logging

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

#: Typesense collection schema dict for the products_v2 (and products) layout.
#: Field order mirrors the live `products` collection as returned by
#: GET /collections/products.
PRODUCTS_SCHEMA_FIELDS: list[dict] = [
    {"name": "product_id", "type": "int32", "sort": True},
    {"name": "name", "type": "string"},
    {"name": "brand", "type": "string", "facet": True},
    {"name": "product_type", "type": "string", "facet": True},
    {"name": "description", "type": "string"},
    {"name": "compatibility_tags", "type": "string[]", "facet": True},
    {"name": "categories", "type": "string[]", "facet": True},
    {"name": "price", "type": "float", "sort": True},
    {
        "name": "embedding",
        "type": "float[]",
        "num_dim": 1024,
        "vec_dist": "cosine",
        "hnsw_params": {"M": 16, "ef_construction": 200},
    },
    {"name": "attributes_json", "type": "string"},
    {"name": "embedding_text", "type": "string"},
    {"name": "image", "type": "string"},
]


def build_collection_schema(name: str) -> dict:
    """Return a Typesense collection creation payload for the given name.

    Args:
        name: Collection name (e.g. 'products_v2').

    Returns:
        Dict suitable for POST /collections body.
    """
    return {
        "name": name,
        "fields": PRODUCTS_SCHEMA_FIELDS,
    }


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


async def ensure_collection(
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    name: str,
    recreate: bool = False,
) -> None:
    """Idempotently create a Typesense collection.

    Behaviour:
    - If the collection does not exist: create it.
    - If the collection exists and recreate=False: do nothing (log info).
    - If the collection exists and recreate=True: delete then create.

    Args:
        http: Async HTTP client.
        base_url: Typesense base URL (e.g. 'http://typesense:8108').
        api_key: Typesense API key.
        name: Collection name to create.
        recreate: When True, delete an existing collection and recreate it.

    Raises:
        httpx.HTTPStatusError: On unexpected Typesense API errors.
    """
    headers = {"X-TYPESENSE-API-KEY": api_key}

    # Check existence
    resp = await http.get(f"{base_url}/collections/{name}", headers=headers)
    exists = resp.status_code == 200

    if exists and not recreate:
        num_docs = resp.json().get("num_documents", "?")
        log.info(
            "collection %r already exists (num_documents=%s), skipping create",
            name,
            num_docs,
        )
        return

    if exists and recreate:
        log.info("recreate=True: deleting collection %r", name)
        del_resp = await http.delete(f"{base_url}/collections/{name}", headers=headers)
        del_resp.raise_for_status()
        log.info("collection %r deleted", name)

    schema = build_collection_schema(name)
    log.info("creating collection %r with %d fields", name, len(schema["fields"]))
    create_resp = await http.post(
        f"{base_url}/collections",
        headers={**headers, "Content-Type": "application/json"},
        content=json.dumps(schema),
    )
    create_resp.raise_for_status()
    log.info("collection %r created successfully", name)
