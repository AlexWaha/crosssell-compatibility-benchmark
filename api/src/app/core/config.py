"""API service settings.

Reads environment variables. API uses four settings groups:
- Catalog DB (read-only catalog reads)
- Metrics DB (read-only recommendations/metrics reads)
- Typesense (search + card image fields)
- Engine URL (search delegate for query embedding)

Each group is documented in the group-ownership matrix (design section 4.4).
DB_PASSWORD is typed SecretStr; callers use .get_secret_value().

Pool instances are created at startup in main.py lifespan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SRC_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = SRC_ROOT / ".env"

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Flat settings for the api service (Catalog DB + Metrics DB + Typesense + Engine)."""

    LOG_LEVEL: str = "INFO"

    # Catalog DB - read-only (products, categories, cards)
    DB_HOST: str = "db"
    DB_USER: str = "user"
    DB_PASSWORD: SecretStr = SecretStr("secret")
    MYSQL_PORT: int = 3306
    DB_CATALOG_DATABASE: str = "avtc_catalog"

    # Metrics DB - read-only (recommendations, summary, metrics)
    DB_METRICS_DATABASE: str = "avtc_metrics"

    # Typesense - search + card image fields
    TYPESENSE_HOST: str = "typesense"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_API_KEY: str = "xyz"
    TYPESENSE_COLLECTION: str = "products"

    # Engine URL - HTTP delegation for query embedding
    ENGINE_URL: str = "http://engine:9000"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def typesense_base_url(self) -> str:
        """Typesense base URL derived from host + port."""
        return f"http://{self.TYPESENSE_HOST}:{self.TYPESENSE_PORT}"


# Module-level singleton used by main.py and repositories
settings = Settings()
