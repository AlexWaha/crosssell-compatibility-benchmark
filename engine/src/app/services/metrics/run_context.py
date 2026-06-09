"""Per-product pipeline run instrumentation.

StageTimer is a context manager that records wall-clock duration and
accumulates per-stage counters (items, errors, llm_calls, tokens, cost).

RunContext wraps a product run: opens a pipeline_runs row on entry,
accumulates StageTimer readings in memory, and flushes everything to
stage_metrics + updates pipeline_runs in a single finish() call.

All DB writes are deferred to finish() so the hot path (retrieval +
verification) is never blocked by instrumentation queries.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from app.db.metrics_repository import MetricsRepository

log = logging.getLogger(__name__)

# Stage names must match the ENUM in the stage_metrics DDL.
_VALID_STAGES = frozenset(
    {
        "ingestion",
        "normalization",
        "embedding",
        "indexing",
        "compatibility",
        "recommendation",
    }
)


class StageTimer:
    """Context manager for timing a single pipeline stage.

    Accumulates counters that can be incremented inside the ``with`` block
    before __exit__ records the elapsed duration.

    Attributes:
        name: Stage name (must be in the stage_metrics ENUM).
        duration_ms: Wall-clock duration in milliseconds (set on __exit__).
        items_processed: Count of items handled in this stage.
        errors: Count of non-fatal errors encountered.
        llm_calls: Number of LLM generate/verify calls made.
        tokens_in: Input tokens consumed across all LLM calls.
        tokens_out: Output tokens produced across all LLM calls.
        cost_usd: Estimated USD cost accumulated in this stage.
        metadata: Arbitrary JSON-serializable dict for extra diagnostics.
    """

    def __init__(self, name: str) -> None:
        if name not in _VALID_STAGES:
            raise ValueError(f"Unknown stage '{name}'. Valid: {sorted(_VALID_STAGES)}")
        self.name: str = name
        self.duration_ms: int = 0
        self.items_processed: int = 0
        self.errors: int = 0
        self.llm_calls: int = 0
        self.tokens_in: int = 0
        self.tokens_out: int = 0
        self.cost_usd: float = 0.0
        self.metadata: dict = {}
        self._start: float = 0.0

    def __enter__(self) -> "StageTimer":
        self._start = time.monotonic()
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.monotonic() - self._start
        self.duration_ms = max(1, int(elapsed * 1000))

    def to_row(self, run_id: int) -> tuple:
        """Build a parameter tuple for the stage_metrics INSERT.

        Column order matches the stage_metrics DDL:
        (run_id, stage, duration_ms, items_processed, errors_count,
         llm_calls, tokens_input, tokens_output, cost_usd, metadata).

        Args:
            run_id: The parent pipeline_runs.run_id.

        Returns:
            Tuple of values ready for executemany.
        """
        meta = json.dumps(self.metadata, ensure_ascii=False) if self.metadata else None
        return (
            run_id,
            self.name,
            self.duration_ms,
            self.items_processed,
            self.errors,
            self.llm_calls,
            self.tokens_in,
            self.tokens_out,
            round(self.cost_usd, 6),
            meta,
        )


class RunContext:
    """Lifecycle manager for one product's pipeline run.

    Usage::

        ctx = RunContext(repo, product_id=42, pipeline_version="1.0.0", mode="jit",
                         experiment_id="baseline_v1")
        run_id = await ctx.open()
        with ctx.stage("compatibility") as st:
            st.llm_calls += 1
            st.items_processed = 5
        await ctx.finish("completed")

    The ``stage()`` context manager returns a StageTimer; the timer is
    automatically appended to the internal list when the ``with`` block exits.

    Args:
        repo: MetricsRepository connected to avtc_metrics.
        product_id: Product being processed.
        pipeline_version: Semantic version string (e.g. "1.0.0").
        mode: Compat mode string ("jit" | "oneshot").
        experiment_id: Experiment identifier written to pipeline_runs for provenance.
    """

    def __init__(
        self,
        repo: "MetricsRepository",
        product_id: int,
        pipeline_version: str = "1.0.0",
        mode: str = "jit",
        experiment_id: str = "baseline_v1",
    ) -> None:
        self._repo = repo
        self._product_id = product_id
        self._pipeline_version = pipeline_version
        self._mode = mode
        self._experiment_id = experiment_id
        self._run_id: int = 0
        self._stages: list[StageTimer] = []

    async def open(self, job_id: int | None = None) -> int:
        """Insert a pipeline_runs row with status='running' and return run_id.

        Args:
            job_id: Optional arq job ID for traceability.

        Returns:
            The newly created run_id.
        """
        self._run_id = await self._repo.start_run(
            product_id=self._product_id,
            pipeline_version=self._pipeline_version,
            job_id=job_id,
            experiment_id=self._experiment_id,
        )
        log.debug(
            "pipeline run opened: run_id=%d product_id=%d mode=%s",
            self._run_id,
            self._product_id,
            self._mode,
        )
        return self._run_id

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTimer]:
        """Context manager yielding a StageTimer for the named stage.

        The timer is appended to the internal accumulator on exit, regardless
        of whether an exception was raised (instrumentation must not suppress
        pipeline errors).

        Args:
            name: Stage name; must be one of the stage_metrics ENUM values.

        Yields:
            StageTimer instance (duration recorded on __exit__).
        """
        timer = StageTimer(name)
        try:
            with timer:
                yield timer
        finally:
            self._stages.append(timer)

    async def finish(self, status: str = "completed", error: str | None = None) -> None:
        """Update pipeline_runs and bulk-insert accumulated stage_metrics.

        Flushes all in-memory stage timers via a single executemany call,
        then updates the pipeline_runs row with final status and total duration.

        Args:
            status: Final status string ("completed" | "failed").
            error: Optional error message stored in error_message column.
        """
        if not self._run_id:
            log.warning(
                "RunContext.finish() called before open() - no run_id, skipping"
            )
            return

        total_ms = sum(s.duration_ms for s in self._stages)

        if self._stages:
            rows = [s.to_row(self._run_id) for s in self._stages]
            await self._repo.write_stage_metrics(self._run_id, rows)

        await self._repo.finish_run(
            run_id=self._run_id,
            status=status,
            total_duration_ms=total_ms,
            error_message=error,
        )
        log.debug(
            "pipeline run finished: run_id=%d status=%s total_ms=%d stages=%d",
            self._run_id,
            status,
            total_ms,
            len(self._stages),
        )
