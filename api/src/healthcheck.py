"""Health check script for the api container.

Checks: both MySQL pools reachable (SELECT 1 on catalog + metrics).
The api does not require engine to be up (degrades to keyword-only search),
so engine is not a health dependency here.

Exit 0 = healthy, non-zero = unhealthy.

Usage (in compose healthcheck block):
    python healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys


async def _check() -> bool:
    """Check catalog and metrics DB connectivity. Return True if both reachable."""
    try:
        from app.core.config import settings
        from app.db.adapter import MySQLAdapter

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
