"""Response/result models for the worker service.

These describe the return values of arq task functions (stored as job results
in Redis by arq). They are informational - arq callers inspect them via job.result().
"""

from __future__ import annotations

from pydantic import BaseModel


class ProcessProductResult(BaseModel):
    """Result of a single process_product_task job.

    This is a passthrough of whatever the engine returns from POST /process-product.

    Args:
        product_id: The product that was processed.
        status: Engine-reported status string (e.g. "ok", "skipped").
        written: Number of recommendations written (when engine reports it).
        raw: Full engine response dict for debugging.
    """

    product_id: int
    status: str = "ok"
    written: int | None = None
    raw: dict | None = None


class RunExperimentResult(BaseModel):
    """Result of a run_experiment_task job (fan-out summary).

    Args:
        experiment_id: The experiment tag used for this run.
        enqueued: Number of process_product_task jobs enqueued.
    """

    experiment_id: str
    enqueued: int
