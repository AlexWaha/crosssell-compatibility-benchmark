"""Tests for StageTimer and RunContext instrumentation.

All tests use an in-memory stub repository - no DB, no network, $0.
"""

from __future__ import annotations

import asyncio

import pytest

from app.services.metrics.run_context import RunContext, StageTimer


# ---------------------------------------------------------------------------
# Stub MetricsRepository (in-memory, no DB)
# ---------------------------------------------------------------------------


class StubRepo:
    """Captures calls made by RunContext for assertion."""

    def __init__(self) -> None:
        self.started: list[dict] = []
        self.finished: list[dict] = []
        self.stage_rows: list[list[tuple]] = []
        self._next_run_id = 1

    async def start_run(
        self, product_id, pipeline_version, job_id=None, experiment_id="baseline_v1"
    ):
        run_id = self._next_run_id
        self._next_run_id += 1
        self.started.append(
            {
                "run_id": run_id,
                "product_id": product_id,
                "pipeline_version": pipeline_version,
                "job_id": job_id,
                "experiment_id": experiment_id,
            }
        )
        return run_id

    async def finish_run(self, run_id, status, total_duration_ms, error_message=None):
        self.finished.append(
            {
                "run_id": run_id,
                "status": status,
                "total_duration_ms": total_duration_ms,
                "error_message": error_message,
            }
        )

    async def write_stage_metrics(self, run_id, rows):
        self.stage_rows.append(rows)


# ---------------------------------------------------------------------------
# StageTimer tests
# ---------------------------------------------------------------------------


def test_stage_timer_unknown_name_raises():
    """StageTimer must reject invalid stage names."""
    with pytest.raises(ValueError, match="Unknown stage"):
        StageTimer("nonexistent_stage")


def test_stage_timer_records_duration():
    """StageTimer records a positive duration_ms after context exit."""
    timer = StageTimer("compatibility")
    with timer:
        pass  # near-instant
    assert timer.duration_ms >= 1


def test_stage_timer_accumulates_counters():
    """Counters set inside the block are preserved after exit."""
    timer = StageTimer("compatibility")
    with timer as t:
        t.items_processed = 42
        t.llm_calls = 3
        t.tokens_in = 100
        t.tokens_out = 200
        t.errors = 1

    assert timer.items_processed == 42
    assert timer.llm_calls == 3
    assert timer.tokens_in == 100
    assert timer.tokens_out == 200
    assert timer.errors == 1


def test_stage_timer_to_row_column_order():
    """to_row() returns a 10-tuple in the expected column order."""
    timer = StageTimer("ingestion")
    with timer as t:
        t.items_processed = 5
        t.llm_calls = 1
        t.tokens_in = 50
        t.tokens_out = 100
        t.cost_usd = 0.001
        t.metadata = {"batch": 1}

    row = timer.to_row(run_id=7)
    assert len(row) == 10
    assert row[0] == 7  # run_id
    assert row[1] == "ingestion"  # stage
    assert row[2] >= 1  # duration_ms
    assert row[3] == 5  # items_processed
    assert row[4] == 0  # errors_count
    assert row[5] == 1  # llm_calls
    assert row[6] == 50  # tokens_input
    assert row[7] == 100  # tokens_output
    assert row[8] == 0.001  # cost_usd (rounded to 6dp)
    assert row[9] is not None  # metadata JSON


def test_stage_timer_null_metadata_when_empty():
    """Empty metadata dict produces NULL (None) in the metadata column."""
    timer = StageTimer("normalization")
    with timer:
        pass
    row = timer.to_row(run_id=1)
    assert row[9] is None


# ---------------------------------------------------------------------------
# RunContext tests
# ---------------------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def test_run_context_open_creates_pipeline_run():
    """open() inserts a pipeline_runs row and returns run_id."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=100, pipeline_version="1.0.0", mode="jit")
    run_id = _run(ctx.open())

    assert run_id == 1
    assert len(repo.started) == 1
    assert repo.started[0]["product_id"] == 100
    assert repo.started[0]["pipeline_version"] == "1.0.0"


def test_run_context_finish_flushes_stage_metrics():
    """finish() writes accumulated stage timers and updates pipeline_runs."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=200, pipeline_version="1.0.0", mode="jit")
    _run(ctx.open())

    with ctx.stage("compatibility") as st:
        st.items_processed = 10

    with ctx.stage("recommendation") as st:
        st.llm_calls = 2

    _run(ctx.finish("completed"))

    # Two stage rows flushed.
    assert len(repo.stage_rows) == 1  # single executemany call
    rows = repo.stage_rows[0]
    assert len(rows) == 2

    # pipeline_runs updated.
    assert len(repo.finished) == 1
    fin = repo.finished[0]
    assert fin["status"] == "completed"
    assert fin["total_duration_ms"] >= 2  # at least 1ms per stage


def test_run_context_status_transitions():
    """finish() with 'failed' passes error message to repo."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=300)
    _run(ctx.open())
    _run(ctx.finish("failed", error="test error"))

    fin = repo.finished[0]
    assert fin["status"] == "failed"
    assert fin["error_message"] == "test error"


def test_run_context_finish_without_open_is_safe():
    """finish() without open() logs a warning and returns gracefully."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=999)
    _run(ctx.finish("completed"))  # should not raise
    assert len(repo.finished) == 0  # nothing written


def test_run_context_no_stages_is_valid():
    """open() then finish() with zero stages is valid (empty stage list)."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=50)
    _run(ctx.open())
    _run(ctx.finish("completed"))

    assert len(repo.finished) == 1
    assert len(repo.stage_rows) == 0  # no stages -> no executemany call


def test_run_context_total_duration_sums_stages():
    """total_duration_ms equals sum of all stage durations."""
    repo = StubRepo()
    ctx = RunContext(repo, product_id=77)
    _run(ctx.open())

    with ctx.stage("ingestion"):
        pass
    with ctx.stage("normalization"):
        pass

    _run(ctx.finish("completed"))

    total = repo.finished[0]["total_duration_ms"]
    rows = repo.stage_rows[0]
    stage_sum = sum(r[2] for r in rows)  # duration_ms is index 2
    assert total == stage_sum
