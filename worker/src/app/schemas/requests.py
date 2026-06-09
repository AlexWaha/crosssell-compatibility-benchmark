"""Outbound request models for the worker service.

These models define the JSON body shapes sent to the engine service over HTTP.
ProcessProductMessage mirrors engine's ProcessReq wire contract - this is
deliberate documented duplication (no shared lib; see design section 6 and ADR-001).
If engine's ProcessReq changes, update this model to match.
"""

from __future__ import annotations

from pydantic import BaseModel


class ProcessProductMessage(BaseModel):
    """Body sent to POST /process-product on the engine service.

    Mirrors engine's ProcessReq (product_id, experiment_id).
    Documented wire-contract duplication - see design ADR-001.

    Args:
        product_id: The product to process.
        experiment_id: Optional experiment tag for result grouping.
    """

    product_id: int
    experiment_id: str | None = None


class RunExperimentMessage(BaseModel):
    """Arguments for the run_experiment_task job.

    Args:
        experiment_id: Tag for this experiment run.
        strategy: Retrieval strategy hint passed through to engine (e.g. "semantic").
    """

    experiment_id: str
    strategy: str | None = None
