"""Engine HTTP response models.

All API responses pass through these DTOs. JSON shape matches the previous
dict-based responses exactly (keys and nesting unchanged) so the worker and api
remain compatible without any client-side changes.
"""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str
    llm_mode: str


class ProcessResult(BaseModel):
    """Response for POST /process-product.

    The product_id and status fields are always present. candidates and written
    are omitted when status indicates no work was done (no_source, no_candidates).
    """

    product_id: int
    status: str
    candidates: int | None = None
    written: int | None = None


class VerifyResult(BaseModel):
    """Response for POST /verify."""

    results: list[dict]
    tokens_in: int
    tokens_out: int


class EmbedResult(BaseModel):
    """Response for POST /embed."""

    embedding: list[float]
