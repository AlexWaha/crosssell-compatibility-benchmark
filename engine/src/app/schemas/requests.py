"""Engine HTTP request models (cross-service wire contract).

These models define the inbound contract for engine endpoints. The worker and api
POST to these endpoints; field names/types must stay stable (worker mirrors
ProcessProductMessage to match ProcessReq).
"""

from __future__ import annotations

from pydantic import BaseModel


class ProcessReq(BaseModel):
    """Request body for POST /process-product.

    Args:
        product_id: ID of the product to process.
        experiment_id: Experiment identifier; defaults to settings.EXPERIMENT_ID if None.
        strategy: Retrieval strategy; defaults to settings.RETRIEVAL_STRATEGY if None.
    """

    product_id: int
    experiment_id: str | None = None
    strategy: str | None = None


class VerifyReq(BaseModel):
    """Request body for POST /verify (direct batch verify, used by tests/inspection).

    Args:
        source: Source product dict.
        candidates: List of candidate product dicts.
    """

    source: dict
    candidates: list[dict]


class EmbedReq(BaseModel):
    """Request body for POST /embed (query embedding delegate from api /search).

    Args:
        text: Text to embed.
    """

    text: str
