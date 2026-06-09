"""Tests for worker task functions.

Coverage:
- run_experiment_task enqueues N jobs (one per active product)
- process_product_task raises RuntimeError on engine 5xx (retryable)
- process_product_task returns engine JSON on 2xx
- POST body shape matches ProcessProductMessage (product_id + experiment_id)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# process_product_task tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_product_task_returns_on_2xx(arq_ctx):
    """process_product_task returns engine JSON on a successful 2xx response."""
    engine_payload = {"product_id": 42, "status": "ok", "written": 3}

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = engine_payload

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.tasks.httpx.AsyncClient", return_value=mock_client):
        from app.services.tasks import process_product_task

        result = await process_product_task(
            arq_ctx, product_id=42, experiment_id="baseline_v1"
        )

    assert result == engine_payload


@pytest.mark.asyncio
async def test_process_product_task_raises_on_5xx(arq_ctx):
    """process_product_task raises RuntimeError on engine 5xx (retryable by arq)."""
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.tasks.httpx.AsyncClient", return_value=mock_client):
        from app.services.tasks import process_product_task

        with pytest.raises(RuntimeError, match="engine 503"):
            await process_product_task(arq_ctx, product_id=99, experiment_id=None)


@pytest.mark.asyncio
async def test_process_product_task_raises_on_500(arq_ctx):
    """process_product_task raises RuntimeError on engine 500."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.tasks.httpx.AsyncClient", return_value=mock_client):
        from app.services.tasks import process_product_task

        with pytest.raises(RuntimeError, match="engine 500"):
            await process_product_task(arq_ctx, product_id=7, experiment_id="exp")


@pytest.mark.asyncio
async def test_process_product_task_post_body_shape(arq_ctx):
    """POST body sent to engine matches ProcessProductMessage (product_id + experiment_id)."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"product_id": 42, "status": "ok"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.tasks.httpx.AsyncClient", return_value=mock_client):
        from app.services.tasks import process_product_task

        await process_product_task(arq_ctx, product_id=42, experiment_id="baseline_v1")

    # Verify the exact JSON body sent to engine
    posted_json = mock_client.post.call_args.kwargs["json"]
    assert posted_json["product_id"] == 42
    assert posted_json["experiment_id"] == "baseline_v1"


@pytest.mark.asyncio
async def test_process_product_task_post_body_none_experiment(arq_ctx):
    """POST body contains experiment_id=None when not supplied."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"product_id": 5, "status": "ok"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("app.services.tasks.httpx.AsyncClient", return_value=mock_client):
        from app.services.tasks import process_product_task

        await process_product_task(arq_ctx, product_id=5)

    call_kwargs = mock_client.post.call_args
    posted_json = call_kwargs.kwargs.get("json", {})
    assert posted_json["product_id"] == 5
    assert posted_json.get("experiment_id") is None


# ---------------------------------------------------------------------------
# run_experiment_task tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_task_enqueues_n_jobs(arq_ctx, mock_catalog_repo):
    """run_experiment_task enqueues exactly N jobs (one per active product ID)."""
    # mock_catalog_repo returns [1, 2, 3] - three active products

    mock_adapter_instance = MagicMock()
    mock_adapter_instance.connect = AsyncMock()
    mock_adapter_instance.close = AsyncMock()

    with (
        patch("app.services.tasks.MySQLAdapter", return_value=mock_adapter_instance),
        patch("app.services.tasks.CatalogRepository", return_value=mock_catalog_repo),
    ):
        from app.services.tasks import run_experiment_task

        result = await run_experiment_task(arq_ctx, experiment_id="baseline_v1")

    assert result["experiment_id"] == "baseline_v1"
    assert result["enqueued"] == 3
    # Verify one enqueue_job call per product ID
    assert arq_ctx["redis"].enqueue_job.call_count == 3


@pytest.mark.asyncio
async def test_run_experiment_task_enqueues_correct_ids(arq_ctx, mock_catalog_repo):
    """run_experiment_task enqueues jobs with the correct product_id and experiment_id."""
    mock_adapter_instance = MagicMock()
    mock_adapter_instance.connect = AsyncMock()
    mock_adapter_instance.close = AsyncMock()

    with (
        patch("app.services.tasks.MySQLAdapter", return_value=mock_adapter_instance),
        patch("app.services.tasks.CatalogRepository", return_value=mock_catalog_repo),
    ):
        from app.services.tasks import run_experiment_task

        await run_experiment_task(arq_ctx, experiment_id="exp_xyz")

    calls = arq_ctx["redis"].enqueue_job.call_args_list
    enqueued_args = [(c.args[0], c.args[1], c.args[2]) for c in calls]
    assert enqueued_args == [
        ("process_product_task", 1, "exp_xyz"),
        ("process_product_task", 2, "exp_xyz"),
        ("process_product_task", 3, "exp_xyz"),
    ]


@pytest.mark.asyncio
async def test_run_experiment_task_zero_products(arq_ctx):
    """run_experiment_task enqueues 0 jobs when catalog has no active products."""
    empty_repo = AsyncMock()
    empty_repo.active_product_ids.return_value = []

    mock_adapter_instance = MagicMock()
    mock_adapter_instance.connect = AsyncMock()
    mock_adapter_instance.close = AsyncMock()

    with (
        patch("app.services.tasks.MySQLAdapter", return_value=mock_adapter_instance),
        patch("app.services.tasks.CatalogRepository", return_value=empty_repo),
    ):
        from app.services.tasks import run_experiment_task

        result = await run_experiment_task(arq_ctx, experiment_id="empty_exp")

    assert result["enqueued"] == 0
    arq_ctx["redis"].enqueue_job.assert_not_called()
