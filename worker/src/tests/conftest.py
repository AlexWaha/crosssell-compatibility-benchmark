"""pytest configuration and fixtures for the worker service tests.

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
# Must happen before any app import so pydantic-settings picks them up.
os.environ.setdefault("DB_PASSWORD", "test")
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_USER", "test")
os.environ.setdefault("DB_CATALOG_DATABASE", "avtc_catalog")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("REDIS_PORT", "6379")
os.environ.setdefault("ENGINE_URL", "http://localhost:9000")

# Add src to path so `from app...` works when pytest is invoked from src/.
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

import pytest  # noqa: E402


@pytest.fixture
def mock_catalog_repo():
    """Return a mock CatalogRepository with 3 active product IDs."""
    repo = AsyncMock()
    repo.active_product_ids.return_value = [1, 2, 3]
    return repo


@pytest.fixture
def mock_redis():
    """Return a mock arq Redis pool that records enqueue_job calls."""
    redis = AsyncMock()
    redis.enqueue_job = AsyncMock(return_value=MagicMock(job_id="test-job-id"))
    return redis


@pytest.fixture
def arq_ctx(mock_redis):
    """Return a minimal arq context dict with a mock redis pool."""
    return {"redis": mock_redis}
