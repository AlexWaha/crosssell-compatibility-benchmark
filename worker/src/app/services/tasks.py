"""arq task functions for the worker service.

The worker computes nothing. It fans out product IDs from the catalog DB and
delegates each product to the engine service over HTTP. Engine 5xx responses are
raised as RuntimeError so arq retries with exponential backoff (max_tries=5).
After max_tries the job is dead-lettered by arq.

Design contract:
- process_product_task: one HTTP POST to engine /process-product; raise on 5xx
- run_experiment_task: read active product IDs, enqueue one job per ID via ctx["redis"]
"""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings
from app.db.adapter import MySQLAdapter
from app.db.catalog_repository import CatalogRepository

log = logging.getLogger(__name__)


async def process_product_task(
    ctx: dict,
    product_id: int,
    experiment_id: str | None = None,
    strategy: str | None = None,
) -> dict:
    """Delegate one product to the engine. Raise on 5xx so arq retries.

    This is the "monitor until the LLM responds" behaviour for slow/error
    scenarios. After max_tries the job is dead-lettered by arq.

    Args:
        ctx: arq context (contains "redis" pool).
        product_id: ID of the product to process.
        experiment_id: Optional experiment tag passed through to engine.
        strategy: Optional retrieval strategy passed through to engine; when None
            the engine falls back to its configured RETRIEVAL_STRATEGY.

    Returns:
        Engine response JSON dict.

    Raises:
        RuntimeError: On engine 5xx response (retryable by arq).
    """
    body = {
        "product_id": product_id,
        "experiment_id": experiment_id,
        "strategy": strategy,
    }
    async with httpx.AsyncClient(timeout=600.0) as client:
        response = await client.post(
            f"{settings.ENGINE_URL}/process-product",
            json=body,
        )
        if response.status_code >= 500:
            raise RuntimeError(
                f"engine {response.status_code}: {response.text[:120]} (retryable)"
            )
        response.raise_for_status()
        return response.json()


async def run_experiment_task(
    ctx: dict,
    experiment_id: str,
    strategy: str = "semantic",
) -> dict:
    """Fan out: enqueue one process_product_task per active catalog product.

    Opens a short-lived catalog DB adapter, fetches all active product IDs,
    then enqueues one process_product_task per ID via the arq Redis pool in ctx.

    Args:
        ctx: arq context (contains "redis" pool).
        experiment_id: Tag for this experiment run (stored in engine recommendations).
        strategy: Retrieval strategy hint (passed through to each product task).

    Returns:
        Dict with experiment_id and enqueued count.
    """
    adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    await adapter.connect()
    try:
        repo = CatalogRepository(db=adapter)
        product_ids = await repo.active_product_ids()
    finally:
        await adapter.close()

    redis = ctx["redis"]
    for pid in product_ids:
        await redis.enqueue_job("process_product_task", pid, experiment_id, strategy)

    log.info(
        "run_experiment %s strategy=%s: fanned out %d product jobs",
        experiment_id,
        strategy,
        len(product_ids),
    )
    return {"experiment_id": experiment_id, "enqueued": len(product_ids)}
