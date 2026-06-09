"""arq WorkerSettings entrypoint.

Run with: arq worker_settings.WorkerSettings

WorkerSettings wires arq to the task functions and Redis broker.
max_tries=5 implements the "monitor until the LLM responds" retry semantics;
after 5 failures arq dead-letters the job (result stored as failure in Redis).
"""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import settings
from app.core.logging import setup_logging
from app.services.tasks import process_product_task, run_experiment_task

setup_logging(settings.LOG_LEVEL)


def _redis_settings() -> RedisSettings:
    """Build arq RedisSettings from the worker Settings object."""
    return RedisSettings(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        database=settings.REDIS_DB,
    )


class WorkerSettings:
    """arq entrypoint: arq worker_settings.WorkerSettings."""

    functions = [process_product_task, run_experiment_task]
    redis_settings = _redis_settings()
    max_tries = 5  # retries before dead-letter - "monitor until LLM responds"
    job_timeout = 900
    max_jobs = 16
