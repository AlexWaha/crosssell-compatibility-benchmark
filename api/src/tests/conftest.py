"""pytest configuration and fixtures for the api service tests.

Adds src/ to sys.path so that `from app...` imports work when running pytest
from the src/ directory or via `python -m pytest tests/`.
Sets required env vars before importing Settings so no .env file is needed in CI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure required env vars are present before Settings is imported.
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_CATALOG_DATABASE", "avtc_catalog")
os.environ.setdefault("DB_METRICS_DATABASE", "avtc_metrics")
os.environ.setdefault("TYPESENSE_HOST", "localhost")
os.environ.setdefault("TYPESENSE_API_KEY", "xyz")
os.environ.setdefault("ENGINE_URL", "http://localhost:9000")

# Add src to path so `from app...` works when pytest is invoked from src/.
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import pytest  # noqa: E402


@pytest.fixture
def mock_catalog_repo() -> AsyncMock:
    """Return a mock CatalogRepository with minimal data."""
    repo = AsyncMock()
    repo.get_categories.return_value = [
        {
            "id": 1,
            "parent_id": 0,
            "name": "Test Cat",
            "slug": "test-cat",
            "product_count": 5,
        }
    ]
    repo.get_product.return_value = {
        "id": 1,
        "slug": "test-product",
        "name": "Test Product",
        "brand": "Brand",
        "product_type": "Widget",
        "price": 9.99,
        "currency": "USD",
        "image": "test.jpg",
        "description": "A test product.",
        "attributes": {},
        "compatibility_tags": [],
        "category_path": [],
    }
    repo.get_cards_by_ids.return_value = {}
    repo.catalog_counts.return_value = (100, 10, 20)
    return repo


@pytest.fixture
def mock_metrics_repo() -> AsyncMock:
    """Return a mock MetricsReadRepository with empty data."""
    repo = AsyncMock()
    repo.top_recommended.return_value = []
    repo.recommendation_rows.return_value = []
    repo.summary.return_value = {
        "total": 0,
        "v1": 0,
        "evaluated": 0,
        "with_reco": 0,
        "asem": None,
        "alog": None,
        "ahyb": None,
        "ctx": [],
    }
    repo.metrics_snapshot.return_value = {
        "with_reco": 0,
        "ctx_rows": [],
        "snap": None,
        "alpha_rows": [],
    }
    repo.compare.return_value = []
    return repo


@pytest.fixture
def mock_http() -> MagicMock:
    """Return a mock httpx.AsyncClient that simulates engine unavailability."""
    client = AsyncMock()
    client.post.side_effect = Exception("engine not available")
    return client
