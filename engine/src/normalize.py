"""CLI: normalize products into the Universal Product Specification (UPS / eta operator).

Reads products from avtc_catalog, skips those whose source_hash matches the stored value
(idempotent + resumable), normalizes the rest via the configured normalizer, and writes
results to product_ai_data.

Usage:
    python -m normalize [--limit N] [--concurrency K]

Options:
    --limit N       Process at most N products (default: 0 = all active products).
    --concurrency K Maximum concurrent normalization tasks (default: COMPAT_CONCURRENCY).

Examples:
    # Mock $0 smoke test on 20 products
    LLM_MODE=mock python -m normalize --limit 20

    # Full real run (requires OPENAI_KEY)
    python -m normalize

Idempotency: if a product's source_hash (computed from name, description, brand,
attributes, categories, model, prompt version) matches the stored hash in
product_ai_data, that product is skipped with zero LLM calls.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date

from app.core.config import settings
from app.db.adapter import MySQLAdapter
from app.db.catalog_repository import CatalogRepository
from app.services.normalization.normalizer import (
    compute_source_hash,
    get_normalizer,
    normalize_product,
)
from app.services.normalization.prompt import PROMPT_VERSION

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_name(cfg) -> str:
    """Return the effective model name string for source_hash and model_used."""
    if cfg.LLM_MODE == "mock":
        return "mock"
    return cfg.PRIMARY_MODEL


def _print_summary(
    total: int,
    processed: int,
    skipped: int,
    failed: int,
    total_tokens_in: int,
    total_tokens_out: int,
    total_filled: int,
    total_attrs_before: int,
    elapsed: float,
) -> None:
    """Print a human-readable run summary to stdout.

    Args:
        total: Total products considered.
        processed: Products that were normalized and written.
        skipped: Products skipped because source_hash matched.
        failed: Products that failed normalization (LLM error or parse failure).
        total_tokens_in: Total input tokens consumed.
        total_tokens_out: Total output tokens consumed.
        total_filled: Total RAG-filled attributes across all products.
        total_attrs_before: Total raw attributes seen before normalization.
        elapsed: Wall-clock elapsed seconds.
    """
    fill_rate_pct = (total_filled / max(total_attrs_before, 1)) * 100
    ppm = (processed / max(elapsed, 0.001)) * 60

    print(
        f"""
=============================================================
  AVTC Normalizer - Run Summary
  Date: {date.today()}
=============================================================

--- Configuration ---
  LLM_MODE         : {settings.LLM_MODE}
  Model            : {_model_name(settings)}
  Prompt version   : {PROMPT_VERSION}
  Concurrency      : {settings.COMPAT_CONCURRENCY}

--- Results ---
  Total considered : {total:,}
  Skipped (hash ok): {skipped:,}
  Processed (wrote): {processed:,}
  Failed           : {failed:,}

--- Attribute Fill (RAG) ---
  Total raw attrs  : {total_attrs_before:,}
  Total filled     : {total_filled:,}
  Fill rate        : {fill_rate_pct:.2f}%

--- Tokens ---
  Input tokens     : {total_tokens_in:,}
  Output tokens    : {total_tokens_out:,}
  Total tokens     : {total_tokens_in + total_tokens_out:,}

--- Performance ---
  Elapsed          : {elapsed:.1f}s
  Throughput       : {ppm:.1f} products/min

--- Cost (placeholder - supply real prices) ---
  *** LLM_MODE={settings.LLM_MODE}: no real spend.
  *** For real runs: estimate cost = (tokens_in/1M)*price_in + (tokens_out/1M)*price_out

=============================================================
"""
    )


# ---------------------------------------------------------------------------
# Core async runner
# ---------------------------------------------------------------------------


async def run(limit: int, concurrency: int) -> None:
    """Main normalization loop.

    Args:
        limit: Max products to process (0 = all active).
        concurrency: Max concurrent normalization tasks.
    """
    import time

    # Build DB adapter (catalog only - no metrics DB needed here)
    db = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
        minsize=2,
        maxsize=max(concurrency // 5, 4),
    )
    await db.connect()
    repo = CatalogRepository(db)

    # Build normalizer (mock or real based on LLM_MODE).
    # Real mode requires a live AsyncOpenAI client - create one here so RealLLM.generate
    # can call client.chat.completions.create(). Passing client=None causes an
    # AttributeError on every call, which is caught as LLMError and silently writes a
    # NULL product_ai_data row (product_type=NULL, embedding_text=NULL, 0 tokens).
    openai_client = None
    if settings.LLM_MODE != "mock":
        import openai

        openai_kwargs: dict = {
            "api_key": settings.OPENAI_KEY.get_secret_value(),
        }
        if settings.OPENAI_BASE_URL:
            openai_kwargs["base_url"] = settings.OPENAI_BASE_URL
        openai_client = openai.AsyncOpenAI(**openai_kwargs)

    normalizer = get_normalizer(settings, client=openai_client)
    model = _model_name(settings)

    log.info(
        "normalizer ready mode=%s model=%s prompt=%s concurrency=%d limit=%d",
        settings.LLM_MODE,
        model,
        PROMPT_VERSION,
        concurrency,
        limit,
    )

    try:
        product_ids = await repo.iter_products_for_normalization(limit=limit)
        total = len(product_ids)
        log.info("candidate products: %d", total)

        sem = asyncio.Semaphore(concurrency)

        # Accumulators
        processed = 0
        skipped = 0
        failed = 0
        total_tokens_in = 0
        total_tokens_out = 0
        total_filled = 0
        total_attrs_before = 0

        t_start = time.monotonic()

        async def _process_one(product_id: int) -> None:
            nonlocal processed, skipped, failed
            nonlocal total_tokens_in, total_tokens_out, total_filled, total_attrs_before

            async with sem:
                # Load raw data
                data = await repo.load_raw_product(product_id)
                if data is None:
                    log.warning("product_id=%d not found, skipping", product_id)
                    return

                product = data["product"]
                raw_attrs: dict[str, str] = data["raw_attrs"]
                categories: list[str] = data["categories"]

                # Check source_hash for idempotency
                current_hash = compute_source_hash(
                    product, raw_attrs, categories, model, PROMPT_VERSION
                )
                stored_hash = await repo.get_ai_data_hash(product_id)
                if stored_hash == current_hash:
                    log.debug("product_id=%d hash match, skipping", product_id)
                    skipped += 1
                    return

                # Normalize
                result = await normalize_product(
                    normalizer,
                    product,
                    raw_attrs,
                    categories,
                    model=model,
                    prompt_version=PROMPT_VERSION,
                )

                m = result["metrics"]
                total_tokens_in += m["tokens_input"]
                total_tokens_out += m["tokens_output"]
                total_filled += m["filled"]
                total_attrs_before += m["attrs_before"]

                if not result["success"]:
                    log.warning(
                        "product_id=%d normalization failed (no UPS), writing null row",
                        product_id,
                    )
                    failed += 1
                    # Still write a row with source_hash so we don't retry on next run
                    await repo.upsert_ai_data(
                        {
                            "product_id": product_id,
                            "normalized_json": None,
                            "compatibility_tags": None,
                            "product_type": None,
                            "embedding_text": None,
                            "model_used": model,
                            "source_hash": result["source_hash"],
                            "version": 1,
                        }
                    )
                    return

                await repo.upsert_ai_data(
                    {
                        "product_id": product_id,
                        "normalized_json": result["normalized_json"],
                        "compatibility_tags": result["compatibility_tags"],
                        "product_type": result["product_type"],
                        "embedding_text": result["embedding_text"],
                        "model_used": model,
                        "source_hash": result["source_hash"],
                        "version": 1,
                    }
                )
                processed += 1
                log.debug(
                    "product_id=%d done type=%s filled=%d ti=%d to=%d",
                    product_id,
                    result["product_type"],
                    m["filled"],
                    m["tokens_input"],
                    m["tokens_output"],
                )

        tasks = [_process_one(pid) for pid in product_ids]
        await asyncio.gather(*tasks)

        elapsed = time.monotonic() - t_start

        _print_summary(
            total=total,
            processed=processed,
            skipped=skipped,
            failed=failed,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_filled=total_filled,
            total_attrs_before=total_attrs_before,
            elapsed=elapsed,
        )

    finally:
        await db.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Normalize products into UPS (eta operator) and write to product_ai_data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  LLM_MODE=mock python -m normalize --limit 20\n"
            "  python -m normalize --limit 200\n"
            "  python -m normalize  # all active products\n"
            "\n"
            "Idempotent: products whose source_hash matches the stored value are skipped."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max products to process (default: 0 = all active). Use --limit 20 for smoke tests.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=settings.COMPAT_CONCURRENCY,
        help=f"Max concurrent tasks (default: COMPAT_CONCURRENCY={settings.COMPAT_CONCURRENCY}).",
    )
    ap.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s %(message)s",
    )

    asyncio.run(run(limit=args.limit, concurrency=args.concurrency))


if __name__ == "__main__":
    main()
