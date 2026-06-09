"""CLI: build and manage the products_v2 Typesense collection.

Subcommands:
    init    Create (or recreate) the Typesense collection schema.
    run     Embed normalized products and upsert into Typesense.
    verify  Check num_documents and sample a document to confirm shape.

The target collection defaults to 'products_v2' so this CLI NEVER touches the
live 'products' collection (which keeps the SPA running). Switch
TYPESENSE_COLLECTION in config once products_v2 is verified.

Usage:
    # Create the collection (idempotent)
    EMBED_PROVIDER=mock python -m index_products init --collection products_v2

    # Index 20 products using MockEmbedder ($0, CI smoke)
    EMBED_PROVIDER=mock python -m index_products run --collection products_v2 --limit 20

    # Verify: num_documents + sample doc shape
    python -m index_products verify --collection products_v2

    # Real OpenAI embed (requires OPENAI_KEY, CEO authorization)
    python -m index_products run --collection products_v2

    # Recreate (wipe + rebuild)
    EMBED_PROVIDER=mock python -m index_products init --collection products_v2 --recreate
    EMBED_PROVIDER=mock python -m index_products run --collection products_v2 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

import httpx

from app.core.config import settings
from app.db.adapter import MySQLAdapter
from app.services.indexing.indexer import run_indexing
from app.services.indexing.schema import ensure_collection

log = logging.getLogger(__name__)

# Default target collection - separate from live 'products' to avoid downtime.
_DEFAULT_COLLECTION = "products_v2"


# ---------------------------------------------------------------------------
# Subcommand: init
# ---------------------------------------------------------------------------


async def _cmd_init(collection: str, recreate: bool) -> None:
    """Create (or recreate) the Typesense collection schema.

    Args:
        collection: Collection name to create.
        recreate: When True, delete the existing collection first.
    """
    async with httpx.AsyncClient(timeout=30.0) as http:
        await ensure_collection(
            http=http,
            base_url=settings.typesense_base_url,
            api_key=settings.TYPESENSE_API_KEY,
            name=collection,
            recreate=recreate,
        )
    print(f"Collection '{collection}' ready.")


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------


async def _cmd_run(collection: str, limit: int, batch_size: int) -> None:
    """Embed normalized products and upsert into Typesense.

    Reads product_ai_data JOIN products, embeds embedding_text via the
    configured EMBED_PROVIDER, and upserts in batches.

    Args:
        collection: Target Typesense collection name.
        limit: Max products to index (0 = all with non-null embedding_text).
        batch_size: Documents per Typesense upsert call.
    """
    db = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
        minsize=2,
        maxsize=10,
    )
    await db.connect()

    try:
        async with httpx.AsyncClient(timeout=60.0) as http:
            stats = await run_indexing(
                cfg=settings,
                db=db,
                http=http,
                collection=collection,
                limit=limit,
                batch_size=batch_size,
            )
    finally:
        await db.close()

    print(
        f"\n=== Indexing complete ===\n"
        f"  Collection : {collection}\n"
        f"  EMBED_PROVIDER : {settings.EMBED_PROVIDER}\n"
        f"  Products loaded: {stats['total']}\n"
        f"  Upserted       : {stats['upserted']}\n"
        f"  Failed         : {stats['failed']}\n"
        f"  Batches        : {stats['batches']}\n"
    )


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------


async def _cmd_verify(collection: str) -> None:
    """Verify the collection: num_documents, sample document shape.

    Args:
        collection: Collection name to inspect.
    """
    headers = {"X-TYPESENSE-API-KEY": settings.TYPESENSE_API_KEY}
    base_url = settings.typesense_base_url

    async with httpx.AsyncClient(timeout=30.0) as http:
        # Get collection metadata
        resp = await http.get(f"{base_url}/collections/{collection}", headers=headers)
        if resp.status_code == 404:
            print(f"Collection '{collection}' does not exist. Run 'init' first.")
            return
        resp.raise_for_status()
        meta = resp.json()
        num_docs = meta.get("num_documents", 0)
        print(f"\n=== Collection: {collection} ===")
        print(f"  num_documents: {num_docs}")

        if num_docs == 0:
            print("  (empty - run 'run' subcommand to index products)")
            return

        # Fetch one document to verify shape
        search_resp = await http.get(
            f"{base_url}/collections/{collection}/documents/search",
            params={"q": "*", "query_by": "name", "per_page": 1},
            headers=headers,
        )
        search_resp.raise_for_status()
        hits = search_resp.json().get("hits", [])
        if not hits:
            print("  No hits returned from search.")
            return

        doc = hits[0]["document"]
        embedding = doc.get("embedding", [])
        embedding_len = len(embedding)

        # Parse attributes_json
        attrs_json_str = doc.get("attributes_json", "{}")
        try:
            attrs_parsed = json.loads(attrs_json_str)
            attrs_ok = isinstance(attrs_parsed, dict)
        except (json.JSONDecodeError, TypeError):
            attrs_ok = False

        print(f"\n  Sample document (id={doc.get('id')}):")
        print(f"    product_id      : {doc.get('product_id')}")
        print(f"    name            : {doc.get('name', '')[:60]}")
        print(f"    brand           : {doc.get('brand')}")
        print(f"    product_type    : {doc.get('product_type')}")
        print(f"    categories      : {doc.get('categories')}")
        print(
            f"    compatibility_tags (first 3): {doc.get('compatibility_tags', [])[:3]}"
        )
        print(f"    price           : {doc.get('price')}")
        print(
            f"    embedding len   : {embedding_len} {'OK' if embedding_len == 1024 else 'MISMATCH'}"
        )
        print(
            f"    attributes_json : parseable={attrs_ok} keys={list(attrs_parsed.keys())[:5]}"
        )
        print(f"    embedding_text  : {(doc.get('embedding_text') or '')[:60]}...")
        print()

        # Assertions
        assert embedding_len == 1024, f"embedding len={embedding_len}, expected 1024"
        assert attrs_ok, "attributes_json is not valid JSON"
        assert isinstance(doc.get("categories"), list), "categories is not a list"
        print("  All shape assertions passed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point with subcommands init / run / verify."""
    ap = argparse.ArgumentParser(
        description="AVTC product indexer - build and manage the products_v2 Typesense collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  EMBED_PROVIDER=mock python -m index_products init --collection products_v2\n"
            "  EMBED_PROVIDER=mock python -m index_products run --collection products_v2 --limit 20\n"
            "  python -m index_products verify --collection products_v2\n"
        ),
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser(
        "init", help="Create (or recreate) the Typesense collection."
    )
    p_init.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=f"Target collection name (default: {_DEFAULT_COLLECTION}).",
    )
    p_init.add_argument(
        "--recreate",
        action="store_true",
        help="Delete the existing collection and recreate it (WARNING: data loss).",
    )

    # run
    p_run = sub.add_parser("run", help="Embed and upsert normalized products.")
    p_run.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=f"Target collection name (default: {_DEFAULT_COLLECTION}).",
    )
    p_run.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max products to index (default: 0 = all with non-null embedding_text).",
    )
    p_run.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Documents per Typesense upsert call (default: 50).",
    )

    # verify
    p_verify = sub.add_parser(
        "verify", help="Verify num_documents and sample doc shape."
    )
    p_verify.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=f"Target collection name (default: {_DEFAULT_COLLECTION}).",
    )

    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    if args.command == "init":
        asyncio.run(_cmd_init(args.collection, args.recreate))
    elif args.command == "run":
        asyncio.run(_cmd_run(args.collection, args.limit, args.batch_size))
    elif args.command == "verify":
        asyncio.run(_cmd_verify(args.collection))


if __name__ == "__main__":
    main()
