"""Product indexer: embed + upsert into Typesense.

Reads normalized products from product_ai_data joined to products and categories,
generates embeddings (MockEmbedder / OpenAI / local Qwen3), builds Typesense
documents in the exact shape that candidates.py expects, and upserts them in
batches via the Typesense import API.

Embedding provider is selected by EMBED_PROVIDER config:
    - 'mock'   -> MockEmbedder (deterministic unit vectors, $0, no network)
    - 'openai' -> embedder.embed_batch via AsyncOpenAI (real spend)
    - 'local'  -> embedder.embed_batch via local Ollama (no OpenAI spend)

The indexing run is idempotent: Typesense import with action=upsert replaces
existing documents by their string id (str(product_id)).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Protocol

import httpx

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------


class EmbedderProtocol(Protocol):
    """Common interface for all embedder backends."""

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return one vector per text.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        ...


# ---------------------------------------------------------------------------
# MockEmbedder
# ---------------------------------------------------------------------------


def _unit_vector(text: str, dim: int = 1024) -> list[float]:
    """Produce a deterministic unit vector of length dim from text.

    Uses the SHA-256 hash of the text as a seed, then normalises the
    resulting vector so its L2 norm is 1.0. Same text always produces the
    same vector.

    Args:
        text: Input string.
        dim: Vector dimension (default 1024 to match the schema).

    Returns:
        L2-normalised float vector of length dim.
    """
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Extend the 32-byte digest to fill dim dimensions by repeating it.
    repeats = math.ceil(dim / len(digest))
    raw_bytes = (digest * repeats)[:dim]
    # Map bytes to floats in [-1, 1]
    vec = [b / 127.5 - 1.0 for b in raw_bytes]
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec))
    if norm < 1e-9:
        vec[0] = 1.0
        norm = 1.0
    return [v / norm for v in vec]


class MockEmbedder:
    """Deterministic $0 embedder for CI and smoke testing.

    Produces unit vectors of dimension 1024 derived from a SHA-256 hash
    of the input text. The same text always produces the same vector
    (deterministic), and the vector has unit L2 norm (cosine-compatible).
    Zero real API calls.

    Args:
        dim: Embedding dimension (default 1024).
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic unit vectors for each text.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors (dim=1024) with unit L2 norm.
        """
        return [_unit_vector(t, self._dim) for t in texts]


# ---------------------------------------------------------------------------
# OpenAI / local embedder wrapper
# ---------------------------------------------------------------------------


class OpenAIEmbedder:
    """Wraps embedder.embed_batch for the OpenAI or local Ollama path.

    Args:
        client: AsyncOpenAI client instance.
        model: Embedding model name.
        dimensions: Embedding dimensions (passed to embed_batch).
        send_dims: Whether to send the dimensions kwarg (False for local models
            that reject the OpenAI-specific parameter).
    """

    def __init__(
        self,
        client,
        model: str,
        dimensions: int = 1024,
        send_dims: bool = True,
    ) -> None:
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._send_dims = send_dims

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the OpenAI/local embeddings API.

        Args:
            texts: Texts to embed.

        Returns:
            List of embedding vectors.
        """
        from app.services.llm import embedder as _embedder

        if self._send_dims:
            return await _embedder.embed_batch(
                self._client,
                texts,
                model=self._model,
                dimensions=self._dimensions,
            )
        # Local models (Qwen3 via Ollama): pass dimensions=None via embed() per-item
        # embed_batch would pass dimensions kwarg -> use it with send_dims logic
        # Actually embed_batch in embedder.py always passes dimensions to the API;
        # for local models we rely on the fact that Qwen3 is natively 1024-dim and
        # we need to call without the dimensions kwarg. Do it text-by-text.
        results = []
        for text in texts:
            v = await _embedder.embed(
                self._client, text, model=self._model, dimensions=None
            )
            results.append(v)
        return results


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------


def get_embedder(cfg) -> EmbedderProtocol:
    """Create the correct embedder based on EMBED_PROVIDER config.

    Selects:
        'mock'   -> MockEmbedder (no network, deterministic, $0)
        'openai' -> OpenAIEmbedder pointing at OpenAI API
        'local'  -> OpenAIEmbedder pointing at local Ollama endpoint

    Args:
        cfg: Settings instance (from app.core.config).

    Returns:
        An object implementing EmbedderProtocol.
    """
    provider = (cfg.EMBED_PROVIDER or "openai").lower()

    if provider == "mock":
        log.info("embedder=MockEmbedder dim=%d", cfg.EMBED_DIMS)
        return MockEmbedder(dim=cfg.EMBED_DIMS)

    from openai import AsyncOpenAI

    if provider == "openai":
        base_url = cfg.OPENAI_BASE_URL or None
        client = AsyncOpenAI(
            api_key=cfg.OPENAI_KEY.get_secret_value(),
            base_url=base_url if base_url else None,
        )
        log.info(
            "embedder=OpenAI model=%s dims=%d base_url=%s",
            cfg.EMBED_MODEL,
            cfg.EMBED_DIMS,
            base_url or "(default)",
        )
        return OpenAIEmbedder(
            client=client,
            model=cfg.EMBED_MODEL,
            dimensions=cfg.EMBED_DIMS,
            send_dims=True,
        )

    if provider == "local":
        client = AsyncOpenAI(
            api_key="local",
            base_url=cfg.EMBED_BASE_URL,
        )
        log.info(
            "embedder=local model=%s dims=%d base_url=%s",
            cfg.EMBED_MODEL,
            cfg.EMBED_DIMS,
            cfg.EMBED_BASE_URL,
        )
        return OpenAIEmbedder(
            client=client,
            model=cfg.EMBED_MODEL,
            dimensions=cfg.EMBED_DIMS,
            send_dims=False,
        )

    raise ValueError(
        f"Unknown EMBED_PROVIDER={provider!r}. Valid values: mock, openai, local."
    )


# ---------------------------------------------------------------------------
# Document builder
# ---------------------------------------------------------------------------


async def build_document(
    ai_row: dict,
    product_row: dict,
    categories: list[str],
    embedding: list[float],
) -> dict:
    """Build a Typesense document in the exact candidates.py shape.

    The document layout mirrors the live `products` collection and satisfies
    _normalize_doc / _parse_attributes in candidates.py without modification.

    Args:
        ai_row: Row from product_ai_data (normalized_json, compatibility_tags,
            product_type, embedding_text from the normalizer).
        product_row: Row from products table (product_id, slug, name,
            description, brand, price, image_path).
        categories: Category name strings for this product.
        embedding: Float vector of length 1024.

    Returns:
        Dict ready for Typesense upsert (id is str(product_id)).
    """
    product_id = product_row["product_id"]

    # attributes_json: json.dumps of the normalized_json dict from product_ai_data.
    # candidates._parse_attributes expects a JSON-parseable string.
    normalized_json = ai_row.get("normalized_json") or {}
    if isinstance(normalized_json, str):
        # Already a JSON string (some DB drivers return it as such)
        try:
            normalized_json = json.loads(normalized_json)
        except (json.JSONDecodeError, TypeError):
            normalized_json = {}
    attributes_json = json.dumps(normalized_json, ensure_ascii=False)

    # compatibility_tags: list of strings
    compat_tags_raw = ai_row.get("compatibility_tags") or []
    if isinstance(compat_tags_raw, str):
        try:
            compat_tags_raw = json.loads(compat_tags_raw)
        except (json.JSONDecodeError, TypeError):
            compat_tags_raw = []
    compatibility_tags: list[str] = list(compat_tags_raw) if compat_tags_raw else []

    product_type = ai_row.get("product_type") or ""
    embedding_text = ai_row.get("embedding_text") or ""
    name = product_row.get("name") or ""
    brand = product_row.get("brand") or ""
    description = product_row.get("description") or ""
    price = float(product_row.get("price") or 0.0)
    image = product_row.get("image_path") or ""

    return {
        "id": str(product_id),
        "product_id": int(product_id),
        "name": name,
        "brand": brand,
        "product_type": product_type,
        "description": description,
        "compatibility_tags": compatibility_tags,
        "categories": list(categories),
        "price": price,
        "embedding": embedding,
        "attributes_json": attributes_json,
        "embedding_text": embedding_text,
        "image": image,
    }


# ---------------------------------------------------------------------------
# Repository query helpers (no ORM - raw parameterized SQL via adapter)
# ---------------------------------------------------------------------------


async def _load_normalized_products(db, limit: int) -> list[dict]:
    """Load product_ai_data rows joined to products.

    Only returns rows with a non-null embedding_text (i.e. successfully
    normalized rows from Component A). Results ordered by product_id ASC.

    Args:
        db: MySQLAdapter instance for avtc_catalog.
        limit: Max rows to return (0 = all).

    Returns:
        List of dicts with keys from both product_ai_data and products.
    """
    sql = (
        "SELECT "
        "  p.product_id, p.slug, p.name, p.description, p.brand, "
        "  CAST(p.price AS DOUBLE) AS price, p.image_path, p.status, "
        "  ai.normalized_json, ai.compatibility_tags, ai.product_type, "
        "  ai.embedding_text "
        "FROM `product_ai_data` ai "
        "INNER JOIN `products` p ON p.product_id = ai.product_id "
        "WHERE ai.embedding_text IS NOT NULL "
        "  AND ai.embedding_text != '' "
        "ORDER BY p.product_id ASC"
    )
    params: tuple = ()
    if limit > 0:
        sql += " LIMIT %s"
        params = (limit,)

    async with db.cursor(dict_cursor=True) as cur:
        await cur.execute(sql, params)
        rows = await cur.fetchall()

    return [dict(r) for r in rows]


async def _load_categories_for_product(db, product_id: int) -> list[str]:
    """Return category names for a product.

    Args:
        db: MySQLAdapter for avtc_catalog.
        product_id: Product ID to look up.

    Returns:
        List of category name strings.
    """
    async with db.cursor(dict_cursor=False) as cur:
        await cur.execute(
            "SELECT c.name "
            "FROM `categories` c "
            "INNER JOIN `product_categories` pc ON pc.category_id = c.category_id "
            "WHERE pc.product_id = %s",
            (product_id,),
        )
        rows = await cur.fetchall()
    return [row[0] for row in rows]


async def _load_all_categories(db, product_ids: list[int]) -> dict[int, list[str]]:
    """Bulk-load category names for a list of product IDs.

    Args:
        db: MySQLAdapter for avtc_catalog.
        product_ids: Product IDs to fetch categories for.

    Returns:
        Dict mapping product_id -> list of category name strings.
    """
    if not product_ids:
        return {}

    placeholders = ",".join(["%s"] * len(product_ids))
    sql = (
        f"SELECT pc.product_id, c.name "
        f"FROM `categories` c "
        f"INNER JOIN `product_categories` pc ON pc.category_id = c.category_id "
        f"WHERE pc.product_id IN ({placeholders})"
    )

    async with db.cursor(dict_cursor=False) as cur:
        await cur.execute(sql, tuple(product_ids))
        rows = await cur.fetchall()

    result: dict[int, list[str]] = {pid: [] for pid in product_ids}
    for product_id, cat_name in rows:
        result.setdefault(int(product_id), []).append(cat_name)
    return result


# ---------------------------------------------------------------------------
# Typesense upsert
# ---------------------------------------------------------------------------


async def _upsert_batch(
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    docs: list[dict],
) -> dict:
    """Upsert a batch of documents via Typesense JSONL import.

    Uses action=upsert so the call is idempotent: existing documents with
    the same id are replaced, new ones are inserted.

    Args:
        http: Async HTTP client.
        base_url: Typesense base URL.
        api_key: Typesense API key.
        collection: Target collection name.
        docs: List of document dicts to upsert.

    Returns:
        Dict with keys 'success' and 'failed' counts.
    """
    jsonl = "\n".join(json.dumps(d, ensure_ascii=False) for d in docs)
    url = f"{base_url}/collections/{collection}/documents/import?action=upsert"
    headers = {
        "X-TYPESENSE-API-KEY": api_key,
        "Content-Type": "text/plain",
    }
    resp = await http.post(url, content=jsonl.encode("utf-8"), headers=headers)
    resp.raise_for_status()

    # Typesense returns one JSON line per document
    success = 0
    failed = 0
    for line in resp.text.strip().splitlines():
        try:
            result = json.loads(line)
            if result.get("success"):
                success += 1
            else:
                failed += 1
                log.warning("upsert failure: %s", result)
        except json.JSONDecodeError:
            log.warning("unparseable import response line: %r", line)
            failed += 1

    return {"success": success, "failed": failed}


# ---------------------------------------------------------------------------
# Main indexing run
# ---------------------------------------------------------------------------


async def run_indexing(
    cfg,
    db,
    http: httpx.AsyncClient,
    collection: str,
    limit: int = 0,
    batch_size: int = 50,
) -> dict:
    """Read normalized products, embed, and upsert into Typesense.

    Steps:
        1. Load normalized product rows from product_ai_data JOIN products.
        2. Bulk-load categories for all products.
        3. Select embedder based on EMBED_PROVIDER.
        4. In batches of batch_size: embed embedding_text, build documents,
           upsert via Typesense import?action=upsert.

    The run is idempotent (upsert by id = str(product_id)) and resumable
    (no state beyond Typesense - rerunning overwrites with the same data).

    Args:
        cfg: Settings instance.
        db: MySQLAdapter connected to avtc_catalog.
        http: Async HTTP client.
        collection: Typesense collection name to index into.
        limit: Max products to index (0 = all with non-null embedding_text).
        batch_size: Documents per Typesense upsert call.

    Returns:
        Dict with keys:
            total (int): Products loaded from DB.
            upserted (int): Documents successfully upserted.
            failed (int): Documents that failed upsert.
            batches (int): Number of Typesense API calls.
    """
    embedder = get_embedder(cfg)
    base_url = cfg.typesense_base_url
    api_key = cfg.TYPESENSE_API_KEY

    log.info(
        "run_indexing collection=%r limit=%d batch_size=%d provider=%s",
        collection,
        limit,
        batch_size,
        cfg.EMBED_PROVIDER,
    )

    rows = await _load_normalized_products(db, limit=limit)
    total = len(rows)
    log.info("loaded %d normalized products from DB", total)

    if total == 0:
        log.warning(
            "no normalized products found in product_ai_data (run normalize.py first)"
        )
        return {"total": 0, "upserted": 0, "failed": 0, "batches": 0}

    product_ids = [int(r["product_id"]) for r in rows]
    cat_map = await _load_all_categories(db, product_ids)
    log.info("loaded categories for %d products", len(cat_map))

    total_upserted = 0
    total_failed = 0
    batches = 0

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        texts = [r.get("embedding_text") or "" for r in batch_rows]

        vectors = await embedder.embed_batch(texts)

        docs = []
        for row, vec in zip(batch_rows, vectors, strict=True):
            pid = int(row["product_id"])
            cats = cat_map.get(pid, [])
            doc = await build_document(
                ai_row=row,
                product_row=row,
                categories=cats,
                embedding=vec,
            )
            docs.append(doc)

        result = await _upsert_batch(http, base_url, api_key, collection, docs)
        total_upserted += result["success"]
        total_failed += result["failed"]
        batches += 1

        log.info(
            "batch %d/%d: upserted=%d failed=%d (total_so_far=%d)",
            batches,
            math.ceil(total / batch_size),
            result["success"],
            result["failed"],
            total_upserted,
        )

    log.info(
        "indexing done total=%d upserted=%d failed=%d batches=%d",
        total,
        total_upserted,
        total_failed,
        batches,
    )
    return {
        "total": total,
        "upserted": total_upserted,
        "failed": total_failed,
        "batches": batches,
    }
