"""Health check script for the worker container.

Checks: Redis broker reachable (PING).
The worker does not require engine to be up (retries on 5xx by design),
so engine is not a health dependency here.

Exit 0 = healthy, non-zero = unhealthy.

Usage (in compose healthcheck block):
    python healthcheck.py
"""

from __future__ import annotations

import asyncio
import sys


async def _check() -> bool:
    """Ping Redis using settings. Return True if reachable."""
    try:
        import redis.asyncio as aioredis

        from app.core.config import settings

        client = aioredis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
        )
        await client.ping()
        await client.aclose()
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
