"""Worker service settings.

Reads environment variables. Worker uses only three settings groups:
- Catalog DB (read-only product-id scan)
- Redis (arq queue broker)
- Engine URL (HTTP delegation target)

Each group is documented in the group-ownership matrix (design section 4.4).
DB_PASSWORD is typed SecretStr; callers use .get_secret_value().

DB adapter instances are created per-call in tasks.py (not cached here).
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
    """Flat settings for the worker service (Catalog DB + Redis + Engine)."""

    LOG_LEVEL: str = "INFO"

    # Catalog DB - read-only (product-id scan for fan-out)
    DB_HOST: str = "db"
    DB_USER: str = "user"
    DB_PASSWORD: SecretStr = SecretStr("secret")
    MYSQL_PORT: int = 3306
    DB_CATALOG_DATABASE: str = "avtc_catalog"

    # Redis - arq queue broker
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Engine URL - HTTP delegation target
    ENGINE_URL: str = "http://engine:9000"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Module-level singleton used by worker_settings.py and tasks.py
settings = Settings()
