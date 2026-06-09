"""Engine HTTP routes (thin handlers).

Each handler: validate request model -> call service -> return response model.
Business logic lives in app.services.*. No inline logic here.

/process-product returns 503 on a retryable LLMError so the worker retries.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.schemas.requests import EmbedReq, ProcessReq, VerifyReq
from app.schemas.responses import (
    EmbedResult,
    HealthResponse,
    ProcessResult,
    VerifyResult,
)
from app.services.llm.embedder import as_query, embed
from app.services.llm.verifier import LLMError, get_llm
from openai import AsyncOpenAI

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness check - returns service status and current LLM mode."""
    settings = request.app.state.settings
    return HealthResponse(status="ok", llm_mode=settings.LLM_MODE)


@router.post("/process-product", response_model=ProcessResult)
async def process(req: ProcessReq, request: Request) -> ProcessResult:
    """Full per-product pipeline. 503 on retryable errors so the worker retries.

    Both LLMError and httpx.HTTPError (ConnectTimeout, ReadTimeout, ConnectError, etc.)
    are mapped to 503 - the worker's "5xx -> retry" logic handles them cleanly.
    """
    from app.services.pipeline import process_product

    settings = request.app.state.settings
    metrics_repo = request.app.state.metrics_repo
    http_client = request.app.state.http
    attr_vocab = getattr(request.app.state, "attr_vocab", {})
    try:
        result = await process_product(
            req.product_id,
            req.experiment_id,
            settings,
            metrics_repo,
            http_client,
            attr_vocab=attr_vocab,
            strategy=req.strategy,
        )
        return ProcessResult(**result)
    except LLMError as exc:
        log.warning("product=%s llm error (retryable): %s", req.product_id, exc)
        raise HTTPException(
            status_code=503, detail=f"llm error (retryable): {exc}"
        ) from exc
    except httpx.HTTPError as exc:
        log.warning(
            "product=%s typesense http error (retryable): %s", req.product_id, exc
        )
        raise HTTPException(
            status_code=503, detail=f"typesense error (retryable): {exc}"
        ) from exc


@router.post("/verify", response_model=VerifyResult)
async def verify(req: VerifyReq, request: Request) -> VerifyResult:
    """Direct single-batch verify (used by tests / inspection)."""
    settings = request.app.state.settings
    client = (
        None
        if settings.LLM_MODE == "mock"
        else AsyncOpenAI(
            api_key=settings.OPENAI_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or None,
        )
    )
    try:
        results, ti, to = await get_llm(settings, client).verify(
            req.source, req.candidates
        )
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return VerifyResult(results=results, tokens_in=ti, tokens_out=to)


@router.post("/embed", response_model=EmbedResult)
async def embed_text(req: EmbedReq, request: Request) -> EmbedResult:
    """Embed a query string in the index's vector space (api /search delegates here)."""
    settings = request.app.state.settings
    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value() or "x",
        base_url=settings.EMBED_BASE_URL or None,
    )
    vec = await embed(client, as_query(req.text), settings.EMBED_MODEL, None)
    return EmbedResult(embedding=vec)
