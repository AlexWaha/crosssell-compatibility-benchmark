"""AVTC storefront BFF (FastAPI). Reads avtc_catalog + avtc_metrics, serves the SPA.

No LLM/embedding compute here: /search delegates query embedding to the engine service.
Entrypoint: uvicorn main:app --host 0.0.0.0 --port 8000
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
from app.db.metrics_repository import MetricsReadRepository
from app.http.routes import router
from app.http.static import mount_static
from app.services.catalog_service import CatalogService
from app.services.metrics_service import MetricsService
from app.services.search_service import SearchService

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: open DB pools + HTTP client on startup, close on shutdown."""
    setup_logging(settings.LOG_LEVEL)

    # Catalog DB adapter
    catalog_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    await catalog_adapter.connect()

    # Metrics DB adapter
    metrics_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_METRICS_DATABASE,
    )
    await metrics_adapter.connect()

    # Repositories
    catalog_repo = CatalogRepository(catalog_adapter)
    metrics_repo = MetricsReadRepository(metrics_adapter)

    # Services
    http_client = httpx.AsyncClient(timeout=60.0)
    catalog_svc = CatalogService(catalog_repo, metrics_repo)
    metrics_svc = MetricsService(catalog_repo, metrics_repo)
    search_svc = SearchService(
        http=http_client,
        engine_url=settings.ENGINE_URL,
        typesense_base_url=settings.typesense_base_url,
        typesense_api_key=settings.TYPESENSE_API_KEY,
        typesense_collection=settings.TYPESENSE_COLLECTION,
    )

    # Stash on app.state for route handlers
    app.state.catalog_repo = catalog_repo
    app.state.metrics_repo = metrics_repo
    app.state.catalog_svc = catalog_svc
    app.state.metrics_svc = metrics_svc
    app.state.search_svc = search_svc

    log.info("BFF ready")
    try:
        yield
    finally:
        await http_client.aclose()
        await catalog_adapter.close()
        await metrics_adapter.close()


app = FastAPI(title="AVTC Storefront BFF", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
mount_static(app)
