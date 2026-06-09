"""Rule cache backends for JIT-ontology rule sets.

Three implementations sharing the RuleCache interface:

- MockRuleCache: in-memory dict, no external deps (tests and $0 CI).
- TableRuleCache: reads/writes the rule_cache MySQL table via MetricsRepository.
- RedisRuleCache: reads/writes Redis (redis.asyncio), backed by the table on miss.

Write-through design (Redis -> table) is implemented in RedisRuleCache when
a MetricsRepository is supplied. Each backend is usable standalone.
"""

from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface (duck-typed, no ABC overhead)
# ---------------------------------------------------------------------------


class RuleCache:
    """Abstract interface for rule caches.

    Concrete backends must implement get and set.
    """

    async def get(self, type_a: str, type_b: str) -> list[dict] | None:
        """Return cached rules or None on miss.

        Args:
            type_a: Source product type.
            type_b: Target product type.

        Returns:
            List of rule dicts or None if not cached.
        """
        raise NotImplementedError

    async def set(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        """Store rules for the given type pair.

        Args:
            type_a: Source product type.
            type_b: Target product type.
            rules: List of rule dicts to cache.
            generated_by: Identifier of the model/source that generated rules.
            source_hash: SHA-256 hex digest of the prompt inputs (for audit).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Mock (in-memory, $0 tests)
# ---------------------------------------------------------------------------


class MockRuleCache(RuleCache):
    """Thread-safe in-memory cache backed by a plain dict.

    Suitable for unit tests and single-process batch runs. State is not
    shared across processes and is lost when the process exits.
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], list[dict]] = {}

    async def get(self, type_a: str, type_b: str) -> list[dict] | None:
        return self._store.get((type_a, type_b))

    async def set(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        self._store[(type_a, type_b)] = rules
        log.debug(
            "mock rule cache set type_a=%s type_b=%s rules=%d",
            type_a,
            type_b,
            len(rules),
        )


# ---------------------------------------------------------------------------
# Table (MySQL rule_cache table via MetricsRepository)
# ---------------------------------------------------------------------------


class TableRuleCache(RuleCache):
    """Durable rule cache backed by the rule_cache MySQL table.

    Provides an auditable record of every generated rule set, required for
    the paper's HDR / evidence-grounding metrics.

    Args:
        repo: MetricsRepository with an open DB connection.
    """

    def __init__(self, repo) -> None:
        # Avoid circular import: repo is typed as Any at runtime
        self._repo = repo

    async def get(self, type_a: str, type_b: str) -> list[dict] | None:
        rules = await self._repo.get_rule_cache(type_a, type_b)
        if rules is not None:
            log.debug(
                "table rule cache hit type_a=%s type_b=%s rules=%d",
                type_a,
                type_b,
                len(rules),
            )
        return rules

    async def set(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        await self._repo.set_rule_cache(
            type_a, type_b, rules, generated_by, source_hash
        )
        log.debug(
            "table rule cache set type_a=%s type_b=%s rules=%d",
            type_a,
            type_b,
            len(rules),
        )


# ---------------------------------------------------------------------------
# Redis (redis.asyncio, write-through to table when repo supplied)
# ---------------------------------------------------------------------------


class RedisRuleCache(RuleCache):
    """Redis-backed rule cache with optional write-through to the MySQL table.

    Key format: ``jit_rules:{type_a}:{type_b}``
    Value: JSON-encoded list of rule dicts.

    When a table_cache is supplied, a cache miss in Redis that results in
    an LLM-generated rule set is also written to the table (write-through).
    Read-path: Redis first, then table_cache on miss, then None.

    Args:
        redis_client: An async redis client (redis.asyncio.Redis).
        ttl_seconds: Redis TTL for cached rule sets (default 7 days).
        table_cache: Optional TableRuleCache for write-through persistence.
    """

    _KEY_PREFIX = "jit_rules"

    def __init__(
        self,
        redis_client,
        ttl_seconds: int = 7 * 24 * 3600,
        table_cache: TableRuleCache | None = None,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl_seconds
        self._table = table_cache

    def _key(self, type_a: str, type_b: str) -> str:
        return f"{self._KEY_PREFIX}:{type_a}:{type_b}"

    async def get(self, type_a: str, type_b: str) -> list[dict] | None:
        raw = await self._redis.get(self._key(type_a, type_b))
        if raw is not None:
            try:
                rules = json.loads(raw)
                log.debug(
                    "redis rule cache hit type_a=%s type_b=%s rules=%d",
                    type_a,
                    type_b,
                    len(rules),
                )
                return rules
            except json.JSONDecodeError:
                log.warning(
                    "redis rule cache corrupt for %s/%s, discarding", type_a, type_b
                )
        # Redis miss - try table cache
        if self._table is not None:
            rules = await self._table.get(type_a, type_b)
            if rules is not None:
                # Back-fill Redis from table
                await self._redis.setex(
                    self._key(type_a, type_b), self._ttl, json.dumps(rules)
                )
                return rules
        return None

    async def set(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        await self._redis.setex(self._key(type_a, type_b), self._ttl, json.dumps(rules))
        log.debug(
            "redis rule cache set type_a=%s type_b=%s rules=%d ttl=%d",
            type_a,
            type_b,
            len(rules),
            self._ttl,
        )
        # Write-through to durable table
        if self._table is not None:
            await self._table.set(type_a, type_b, rules, generated_by, source_hash)
