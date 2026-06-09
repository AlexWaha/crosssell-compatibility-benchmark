"""API routes for the storefront BFF.

All /api/* endpoints live here. Handlers are thin: validate -> service -> return dict.
Business logic lives in services/; SQL lives in db/repositories.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

from app.schemas.requests import SearchRequest

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/health")
async def health() -> dict:
    """Liveness check. Returns {"status": "ok"}."""
    return {"status": "ok"}


@router.get("/api/categories")
async def categories(request: Request) -> dict:
    """Return all active categories with product counts."""
    items = await request.app.state.catalog_repo.get_categories()
    return {"items": items}


@router.get("/api/categories/{ident}/products")
async def category_products(
    ident: str,
    request: Request,
    page: int = Query(1, ge=1),
    page_size: int = Query(24, ge=1, le=100),
) -> dict:
    """Return paginated products for a category (by id or slug).

    Args:
        ident: Category id (numeric string) or slug.
        page: 1-based page number.
        page_size: Number of items per page (1-100).
    """
    cid = await request.app.state.catalog_svc.resolve_id(ident, "category_id")
    if cid is None:
        raise HTTPException(status_code=404, detail="category not found")
    return await request.app.state.catalog_repo.get_category_products(
        cid, page, page_size
    )


@router.get("/api/products/{ident}")
async def product(ident: str, request: Request) -> dict:
    """Return full product detail (by id or slug)."""
    pid = await request.app.state.catalog_svc.resolve_id(ident, "product_id")
    p = (
        await request.app.state.catalog_repo.get_product(pid)
        if pid is not None
        else None
    )
    if not p:
        raise HTTPException(status_code=404, detail="product not found")
    return p


@router.get("/api/products/{ident}/recommendations")
async def recommendations(
    ident: str,
    request: Request,
    limit: int = Query(12, ge=1, le=50),
    experiment: str = Query("baseline_v1"),
) -> dict:
    """Return recommendation items for a product.

    Args:
        ident: Product id (numeric string) or slug.
        limit: Maximum number of recommendations (1-50).
        experiment: Experiment ID to filter by.
    """
    pid = await request.app.state.catalog_svc.resolve_id(ident, "product_id")
    if pid is None:
        raise HTTPException(status_code=404, detail="product not found")
    return await request.app.state.catalog_svc.get_recommendations(
        pid, limit, experiment
    )


@router.get("/api/top-products")
async def top_products(
    request: Request,
    limit: int = Query(20, ge=1, le=50),
    experiment: str = Query("baseline_v1"),
) -> dict:
    """Return top recommended products by recommendation count.

    Args:
        limit: Maximum number of products (1-50).
        experiment: Experiment ID to filter by.
    """
    return await request.app.state.catalog_svc.get_top_products(limit, experiment)


@router.get("/api/summary")
async def summary(
    request: Request,
    experiment: str = Query("baseline_v1"),
) -> dict:
    """Return aggregate summary stats for an experiment."""
    return await request.app.state.metrics_svc.summary(experiment)


@router.get("/api/metrics")
async def metrics_full(
    request: Request,
    experiment: str = Query("baseline_v1"),
) -> dict:
    """Return full quality metrics for an experiment."""
    return await request.app.state.metrics_svc.metrics_full(experiment)


@router.get("/api/compare")
async def compare(
    request: Request,
    experiments: str = Query("baseline_v1,improved_v2"),
) -> dict:
    """Compare multiple experiments.

    Args:
        experiments: Comma-separated list of experiment IDs.
    """
    exp_ids = [e.strip() for e in experiments.split(",") if e.strip()]
    catalog, _c, _b = await request.app.state.catalog_repo.catalog_counts()
    return await request.app.state.metrics_svc.compare(exp_ids, catalog)


@router.post("/api/search")
async def search(req: SearchRequest, request: Request) -> dict:
    """Hybrid search: delegate embedding to engine, query Typesense.

    Falls back to keyword-only if engine is unavailable.
    """
    return await request.app.state.search_svc.search(req.query, req.limit)
