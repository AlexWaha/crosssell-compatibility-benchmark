"""AVTC Engine compute service (internal FastAPI).

All LLM/Typesense/scoring lives here; the worker and api call it over HTTP.
Never serves the storefront.

Entrypoint: uvicorn server:app --host 0.0.0.0 --port 9000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.adapter import MySQLAdapter
from app.db.catalog_repository import CatalogRepository
from app.db.metrics_repository import MetricsRepository
from app.db.schema.metrics import TABLES as METRICS_TABLES, get_create_table_sql
from app.http.routes import router

log = logging.getLogger(__name__)

# Shared Typesense httpx client: sane timeouts, pooled connections.
# 10s connect is generous for a local Typesense container.
_TYPESENSE_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)
_TYPESENSE_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=16)


async def _init_metrics_schema(metrics_adapter: MySQLAdapter) -> None:
    """Idempotent CREATE TABLE IF NOT EXISTS for all avtc_metrics tables.

    Called once at engine startup so the metrics schema always exists without
    requiring a separate manual DDL step. Safe to run on every boot.
    """
    async with metrics_adapter.cursor(dict_cursor=False) as cur:
        for table_name in METRICS_TABLES:
            ddl = get_create_table_sql(table_name)
            await cur.execute(ddl)
    log.info("metrics schema verified (%d tables)", len(METRICS_TABLES))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: open DB pools and shared HTTP client on startup, close on shutdown."""
    setup_logging(settings.LOG_LEVEL)

    # Catalog DB adapter (catalog importer + fan-out helpers)
    catalog_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    await catalog_adapter.connect()

    # Metrics DB adapter (write recommendations/evaluations)
    metrics_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_METRICS_DATABASE,
    )
    await metrics_adapter.connect()
    await _init_metrics_schema(metrics_adapter)

    metrics_repo = MetricsRepository(metrics_adapter)
    catalog_repo = CatalogRepository(catalog_adapter)

    # Shared httpx client for Typesense requests - pooled, sane timeouts.
    # The OpenAI/LLM path uses its own SDK client and is unaffected.
    http_client = httpx.AsyncClient(
        timeout=_TYPESENSE_TIMEOUT,
        limits=_TYPESENSE_LIMITS,
    )

    # Build per-type attribute vocabulary once at startup.
    # Grounding rule-gen prompts in real attribute keys prevents the LLM from
    # inventing keys that never appear in normalized_json (which caused L=0
    # for all candidates: rules_undefined=N/N, zero recommendations).
    try:
        attr_vocab = await catalog_repo.attribute_vocab_by_type(top_n=25)
        log.info("attr_vocab loaded for %d product types", len(attr_vocab))
    except Exception as exc:
        log.warning("attr_vocab load failed (%s), vocab-grounding disabled", exc)
        attr_vocab = {}

    # Stash on app.state for route handlers
    app.state.settings = settings
    app.state.catalog_adapter = catalog_adapter
    app.state.metrics_adapter = metrics_adapter
    app.state.metrics_repo = metrics_repo
    app.state.catalog_repo = catalog_repo
    app.state.attr_vocab = attr_vocab
    app.state.http = http_client

    log.info("engine ready (llm_mode=%s)", settings.LLM_MODE)
    try:
        yield
    finally:
        await http_client.aclose()
        await catalog_adapter.close()
        await metrics_adapter.close()


app = FastAPI(title="AVTC Engine", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
