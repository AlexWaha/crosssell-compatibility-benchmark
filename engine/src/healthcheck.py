"""Health check script for the engine container.

Checks:
- MySQL catalog DB reachable (SELECT 1)
- MySQL metrics DB reachable (SELECT 1)
- Typesense reachable (GET /health via httpx)

Reports llm_mode for observability. Does NOT call the LLM (zero spend).
mock is a valid operational mode - the healthcheck does NOT fail on mock.

Exit 0 = healthy, non-zero = unhealthy.

Usage (in compose healthcheck block):
    python healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys


async def _check() -> bool:
    """Run all dependency checks. Return True if all pass."""
    try:
        from app.core.config import settings
        from app.db.adapter import MySQLAdapter

        # Check catalog DB
        catalog = MySQLAdapter(
            host=settings.DB_HOST,
            port=settings.MYSQL_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD.get_secret_value(),
            db=settings.DB_CATALOG_DATABASE,
        )
        await catalog.connect()
        async with catalog.cursor(dict_cursor=False) as cur:
            await cur.execute("SELECT 1")
        await catalog.close()

        # Check metrics DB
        metrics = MySQLAdapter(
            host=settings.DB_HOST,
            port=settings.MYSQL_PORT,
            user=settings.DB_USER,
            password=settings.DB_PASSWORD.get_secret_value(),
            db=settings.DB_METRICS_DATABASE,
        )
        await metrics.connect()
        async with metrics.cursor(dict_cursor=False) as cur:
            await cur.execute("SELECT 1")
        await metrics.close()

        # Check Typesense
        import httpx

        ts_url = f"http://{settings.TYPESENSE_HOST}:{settings.TYPESENSE_PORT}/health"
        async with httpx.AsyncClient(timeout=5.0) as http:
            resp = await http.get(ts_url)
            resp.raise_for_status()

        # Report llm_mode for observability (mock is valid, not a failure)
        print(f"healthcheck ok: llm_mode={settings.LLM_MODE}", flush=True)
        return True

    except Exception as exc:
        print(f"healthcheck failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    """Run the health check and return exit code."""
    ok = asyncio.run(_check())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
