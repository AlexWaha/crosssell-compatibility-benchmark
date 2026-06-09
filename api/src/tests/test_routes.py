"""Smoke tests for api routes using mocked repositories.

These tests verify that route handlers correctly delegate to services and
return the expected response shapes, without hitting any real DB or network.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

import sys
from pathlib import Path

# Ensure src is on path (conftest also does this, but be explicit for clarity)
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def _make_app(mock_catalog_repo, mock_metrics_repo, mock_http):
    """Build a FastAPI test app with mocked state (bypasses real lifespan)."""
    from app.http.routes import router
    from app.services.catalog_service import CatalogService
    from app.services.metrics_service import MetricsService
    from app.services.search_service import SearchService
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(router)

    catalog_svc = CatalogService(mock_catalog_repo, mock_metrics_repo)
    metrics_svc = MetricsService(mock_catalog_repo, mock_metrics_repo)
    search_svc = SearchService(
        http=mock_http,
        engine_url="http://engine:9000",
        typesense_base_url="http://typesense:8108",
        typesense_api_key="xyz",
        typesense_collection="products",
    )

    app.state.catalog_repo = mock_catalog_repo
    app.state.metrics_repo = mock_metrics_repo
    app.state.catalog_svc = catalog_svc
    app.state.metrics_svc = metrics_svc
    app.state.search_svc = search_svc

    return app


@pytest.fixture
def client(mock_catalog_repo, mock_metrics_repo, mock_http):
    """Return a TestClient with mocked state."""
    app = _make_app(mock_catalog_repo, mock_metrics_repo, mock_http)
    return TestClient(app, raise_server_exceptions=True)


def test_health(client):
    """GET /api/health returns {"status": "ok"}."""
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_categories(client):
    """GET /api/categories returns items list."""
    resp = client.get("/api/categories")
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert isinstance(body["items"], list)
    assert body["items"][0]["name"] == "Test Cat"


def test_product_found(client, mock_catalog_repo):
    """GET /api/products/1 returns product detail when found."""
    mock_catalog_repo.id_by_slug = AsyncMock(return_value=None)
    resp = client.get("/api/products/1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == 1
    assert body["name"] == "Test Product"


def test_product_not_found(client, mock_catalog_repo):
    """GET /api/products/9999 returns 404 when product missing."""
    mock_catalog_repo.get_product = AsyncMock(return_value=None)
    resp = client.get("/api/products/9999")
    assert resp.status_code == 404


def test_recommendations_empty(client, mock_catalog_repo):
    """GET /api/products/1/recommendations returns empty items when no data."""
    resp = client.get("/api/products/1/recommendations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == 1
    assert body["items"] == []


def test_top_products_empty(client):
    """GET /api/top-products returns empty items when no data."""
    resp = client.get("/api/top-products")
    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_summary_shape(client):
    """GET /api/summary returns expected keys."""
    resp = client.get("/api/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert "model" in body
    assert "catalog" in body
    assert "coverage" in body
    assert "by_context" in body


def test_metrics_shape(client):
    """GET /api/metrics returns expected keys."""
    resp = client.get("/api/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "catalog" in body
    assert "pAtK" in body
    assert "alpha" in body
    assert len(body["pAtK"]) == 3


def test_search_keyword_fallback(mock_catalog_repo, mock_metrics_repo):
    """POST /api/search falls back to keyword-only when engine embed fails.

    Uses a custom http mock that:
    - Raises for the engine /embed call (engine unavailable)
    - Returns a valid Typesense response for the multi_search call

    Verifies the fallback path returns 200 with correct shape.
    """
    from unittest.mock import AsyncMock, MagicMock

    # Build a Typesense-like response
    ts_response = MagicMock()
    ts_response.status_code = 200
    ts_response.json.return_value = {"results": [{"hits": []}]}
    ts_response.raise_for_status = MagicMock()

    call_count = 0

    async def _post_side_effect(url, **kwargs):
        nonlocal call_count
        call_count += 1
        if "embed" in url:
            raise Exception("engine not available")
        # Typesense multi_search call
        return ts_response

    http_mock = AsyncMock()
    http_mock.post.side_effect = _post_side_effect

    app = _make_app(mock_catalog_repo, mock_metrics_repo, http_mock)
    test_client = TestClient(app, raise_server_exceptions=True)

    resp = test_client.post("/api/search", json={"query": "iphone", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "iphone"
    assert body["items"] == []
    # Engine call was made (and caught), Typesense call succeeded
    assert call_count == 2


def test_boundary_no_openai_no_arq():
    """Verify that openai and arq are not imported by the api service modules."""
    import sys

    # Import the main module tree
    import app.core.config  # noqa: F401
    import app.db.adapter  # noqa: F401
    import app.db.catalog_repository  # noqa: F401
    import app.db.metrics_repository  # noqa: F401
    import app.http.routes  # noqa: F401
    import app.services.catalog_service  # noqa: F401
    import app.services.metrics_service  # noqa: F401
    import app.services.search_service  # noqa: F401

    assert "openai" not in sys.modules, "openai must not be imported by the api service"
    assert "arq" not in sys.modules, "arq must not be imported by the api service"
    assert "tiktoken" not in sys.modules, (
        "tiktoken must not be imported by the api service"
    )


def test_recommendations_with_data(mock_catalog_repo, mock_metrics_repo):
    """GET /api/products/1/recommendations returns joined items when data exists.

    Verifies that the catalog card fields and score fields are merged correctly
    by CatalogService.get_recommendations (the join logic).
    """
    from unittest.mock import AsyncMock

    mock_metrics_repo.recommendation_rows = AsyncMock(
        return_value=[
            {
                "recommended_id": 42,
                "context_code": "cross",
                "hybrid_score": 0.85,
                "semantic_score": 0.9,
                "logical_score": 0.8,
            }
        ]
    )
    mock_catalog_repo.get_cards_by_ids = AsyncMock(
        return_value={
            42: {
                "id": 42,
                "name": "Rec Product",
                "brand": "Brand",
                "product_type": "Widget",
                "price": 19.99,
                "currency": "USD",
                "image": "image.jpg",
            }
        }
    )

    app = _make_app(mock_catalog_repo, mock_metrics_repo, AsyncMock())
    test_client = TestClient(app, raise_server_exceptions=True)

    resp = test_client.get("/api/products/1/recommendations")
    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == 1
    assert len(body["items"]) == 1
    item = body["items"][0]
    # Catalog card fields must be present
    assert item["id"] == 42
    assert item["name"] == "Rec Product"
    # Score fields must be merged in
    assert item["context_code"] == "cross"
    assert item["hybrid_score"] == pytest.approx(0.85)
    assert item["semantic_score"] == pytest.approx(0.9)
    assert item["logical_score"] == pytest.approx(0.8)


def test_category_products_not_found(client):
    """GET /api/categories/99999/products returns 404 when category missing."""
    from unittest.mock import AsyncMock

    client.app.state.catalog_svc._catalog.id_by_slug = AsyncMock(return_value=None)
    # Use a non-numeric ident so slug lookup is triggered
    resp = client.get("/api/categories/nonexistent-slug/products")
    assert resp.status_code == 404


def test_summary_with_data(mock_catalog_repo, mock_metrics_repo):
    """GET /api/summary returns correctly assembled dict when metrics data exists."""
    from unittest.mock import AsyncMock

    mock_catalog_repo.catalog_counts = AsyncMock(return_value=(100, 10, 20))
    mock_metrics_repo.summary = AsyncMock(
        return_value={
            "total": 50,
            "v1": 40,
            "evaluated": 30,
            "with_reco": 25,
            "asem": 0.75,
            "alog": 0.65,
            "ahyb": 0.70,
            "ctx": [("cross", 20), ("upsell", 5)],
        }
    )

    app = _make_app(mock_catalog_repo, mock_metrics_repo, AsyncMock())
    test_client = TestClient(app, raise_server_exceptions=True)

    resp = test_client.get("/api/summary")
    assert resp.status_code == 200
    body = resp.json()
    # Verify all required keys exist
    for key in (
        "model",
        "catalog",
        "evaluated",
        "with_reco",
        "coverage",
        "total_pairs",
        "verdict1_pairs",
        "verdict0_pairs",
        "verdict1_share",
        "avg_semantic",
        "avg_logical",
        "avg_hybrid",
        "by_context",
    ):
        assert key in body, f"Missing key: {key}"
    # Verify computed values
    assert body["total_pairs"] == 50
    assert body["verdict1_pairs"] == 40
    assert body["verdict0_pairs"] == 10
    assert body["verdict1_share"] == pytest.approx(0.8)
    assert body["coverage"] == pytest.approx(0.25)
    assert body["avg_hybrid"] == pytest.approx(0.70)
    assert len(body["by_context"]) == 2
    assert body["by_context"][0]["code"] == "cross"


def test_compare_empty(client):
    """GET /api/compare returns catalog count and empty experiments list when no data."""
    resp = client.get("/api/compare?experiments=baseline_v1")
    assert resp.status_code == 200
    body = resp.json()
    assert "catalog" in body
    assert "experiments" in body
    assert isinstance(body["experiments"], list)
