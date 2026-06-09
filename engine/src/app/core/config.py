"""Engine service settings.

Reads environment variables. Engine uses all settings groups (it is the compute hub
and schema owner):
- Catalog DB (importer + fan-out helpers)
- Metrics DB (write recommendations/evaluations)
- OpenAI / LLM (verify, embed)
- Typesense (retrieve candidates)
- Compatibility (pipeline parameters)

Each group is documented in the group-ownership matrix (design section 4.4).
OPENAI_KEY and DB_PASSWORD are typed SecretStr; callers use .get_secret_value().

Pool instances are created at startup in server.py lifespan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

SRC_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = SRC_ROOT / ".env"

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Flat settings for the engine service (all groups)."""

    LOG_LEVEL: str = "INFO"

    # Catalog DB - catalog importer + Typesense source fan-out helpers
    DB_HOST: str = "db"
    DB_USER: str = "user"
    DB_PASSWORD: SecretStr = SecretStr("secret")
    MYSQL_PORT: int = 3306
    DB_CATALOG_DATABASE: str = "avtc_catalog"

    # Metrics DB - write recommendations and evaluations
    DB_METRICS_DATABASE: str = "avtc_metrics"

    # OpenAI / LLM - verify and embed
    OPENAI_KEY: SecretStr = SecretStr("")
    OPENAI_BASE_URL: str = ""
    PRIMARY_MODEL: str = "gpt-5-nano"
    JUDGE_MODEL: str = "gpt-5"
    EMBED_MODEL: str = "dengcao/Qwen3-Embedding-0.6B:Q8_0"
    EMBED_BASE_URL: str = "http://host.docker.internal:11434/v1"
    EMBED_DIMS: int = 1024
    MAX_TOKENS: int = 4000
    REASONING_EFFORT: str = "low"
    # "real" hits the provider; "mock" routes to MockLLM ($0 tests / smoke runs).
    LLM_MODE: str = "real"
    # Mock scenario: valid | slow | error | truncated
    MOCK_SCENARIO: str = "valid"
    # Embedding provider: 'openai' (text-embedding-3-large), 'local' (Qwen3 via Ollama),
    # 'mock' (deterministic unit vectors, $0, CI).
    EMBED_PROVIDER: str = "openai"

    # Typesense - vector search for candidates
    TYPESENSE_HOST: str = "typesense"
    TYPESENSE_PORT: int = 8108
    TYPESENSE_API_KEY: str = "xyz"
    # Default collection for live retrieval. Keep 'products' (the existing populated
    # collection) until products_v2 is fully built and verified. The indexer CLI
    # accepts --collection to target products_v2 explicitly, so building v2 does NOT
    # affect the live SPA. Switch this to 'products_v2' after verification.
    TYPESENSE_COLLECTION: str = "products"

    # Compatibility pipeline parameters
    COMPAT_CONCURRENCY: int = Field(default=50, ge=1, le=200)
    COMPAT_BATCH_SIZE: int = Field(default=10, ge=1, le=50)
    COMPAT_TOP_K: int = Field(default=25, ge=5, le=200)
    COMPAT_ALPHA: float = Field(default=0.6, ge=0.0, le=1.0)
    COMPAT_TAU_S: float = Field(default=0.3, ge=0.0, le=1.0)
    COMPAT_TAU_L: float = Field(default=0.5, ge=0.0, le=1.0)
    EXPERIMENT_ID: str = "baseline_v1"
    RETRIEVAL_STRATEGY: str = "semantic"

    # JIT-ontology verification (Component C)
    # COMPAT_MODE: 'jit' uses rule-based L = PROD l_k^w; 'oneshot' uses legacy LLM verify
    COMPAT_MODE: str = "jit"
    # TAU_Q: minimum evidence fraction for HDR attribute gate (0 = gate disabled effectively)
    TAU_Q: float = Field(default=0.5, ge=0.0, le=1.0)
    # HDR_ENABLED: drop rules whose attributes never appear in the product vocab
    HDR_ENABLED: bool = True
    # RULE_CACHE_BACKEND: 'redis' | 'table' | 'mock'
    RULE_CACHE_BACKEND: str = "redis"
    # L_AGG: 'weighted_product' (article default) | 'product' (unweighted, control arm)
    L_AGG: str = "weighted_product"

    # Redis connection (for RedisRuleCache)
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # Optional RPM limiter for OpenAI calls (0 = disabled, no behavior change)
    OPENAI_RPM_LIMIT: int = Field(default=0, ge=0)

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def typesense_base_url(self) -> str:
        """Typesense base URL derived from host + port."""
        return f"http://{self.TYPESENSE_HOST}:{self.TYPESENSE_PORT}"


# Module-level singleton used by server.py, CLI entrypoints, and repositories
settings = Settings()
