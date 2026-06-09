"""CLI to enqueue an arq job.

Usage: python -m enqueue <task_name> [args...]

Numeric args are coerced to int. Used for ops and smoke tests.

Examples:
    python -m enqueue run_experiment_task baseline_v1
    python -m enqueue process_product_task 42
    python -m enqueue process_product_task 42 baseline_v1
"""

from __future__ import annotations

import asyncio
import sys

from arq import create_pool
from arq.connections import RedisSettings

from app.core.config import settings


async def main() -> None:
    """Parse CLI args and enqueue one arq job."""
    if len(sys.argv) < 2:
        print("Usage: python -m enqueue <task_name> [args...]")
        sys.exit(1)

    task = sys.argv[1]
    args = [int(a) if a.lstrip("-").isdigit() else a for a in sys.argv[2:]]

    pool = await create_pool(
        RedisSettings(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            database=settings.REDIS_DB,
        )
    )
    job = await pool.enqueue_job(task, *args)
    print(f"enqueued {task} {args} -> job_id={job.job_id}")


if __name__ == "__main__":
    asyncio.run(main())
