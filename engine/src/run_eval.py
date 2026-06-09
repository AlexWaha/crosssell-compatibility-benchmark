"""CLI: compute quality metrics for a completed (or in-progress) experiment.

Usage:
    python -m run_eval [experiment_id]

If experiment_id is omitted, reads Settings.EXPERIMENT_ID (default: baseline_v1).

This script is $0: it does NOT call OpenAI, Typesense, or any LLM.
It reads from avtc_metrics.compatibility_evaluations (all evaluated pairs) and
ground_truth, then writes to quality_snapshots + alpha_experiments.

Bootstrap CI seed: BOOTSTRAP_SEED=42 (fixed for reproducibility -- see metrics.py).
"""

from __future__ import annotations

import asyncio
import logging
import sys

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.adapter import MySQLAdapter
from app.db.metrics_repository import MetricsRepository
from app.services.eval.evaluator import run_evaluation

log = logging.getLogger(__name__)


async def main(experiment_id: str) -> None:
    """Connect to DB, run evaluation, print summary to stdout."""
    setup_logging(settings.LOG_LEVEL)

    metrics_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_METRICS_DATABASE,
    )
    await metrics_adapter.connect()

    repo = MetricsRepository(metrics_adapter)

    try:
        snapshot, alpha_rows = await run_evaluation(
            experiment_id=experiment_id,
            repo=repo,
            tau_s=settings.COMPAT_TAU_S,
            tau_l=settings.COMPAT_TAU_L,
        )
    finally:
        await metrics_adapter.close()

    # CLI summary (print is intentional here -- this is the CLI output surface)
    print(f"\n=== run_eval complete: experiment={experiment_id} ===")
    print(f"  sample_size  : {snapshot['sample_size']}")
    print(
        f"  TVR          : {snapshot['tvr']:.6f}"
        if snapshot["tvr"] is not None
        else "  TVR          : None"
    )
    print(
        f"  SLD          : {snapshot['sld']:.6f}"
        if snapshot["sld"] is not None
        else "  SLD          : None"
    )
    print(
        f"  HSV          : {snapshot['hsv']:.6f}"
        if snapshot["hsv"] is not None
        else "  HSV          : None"
    )
    print(
        f"  alpha*       : {snapshot['alpha_optimal']:.2f}"
        if snapshot["alpha_optimal"] is not None
        else "  alpha*       : None"
    )
    print(
        f"  precision@10 : {snapshot['precision_at_10']}"
        if snapshot["precision_at_10"] is not None
        else "  precision@10 : None (no ground-truth)"
    )
    print(
        f"  recall@10    : {snapshot['recall_at_10']}"
        if snapshot["recall_at_10"] is not None
        else "  recall@10    : None (no ground-truth)"
    )
    print(
        f"  ndcg@10      : {snapshot['ndcg_at_10']}"
        if snapshot["ndcg_at_10"] is not None
        else "  ndcg@10      : None (no ground-truth)"
    )
    print(f"  alpha_rows   : {len(alpha_rows)}")
    print("  quality_snapshots: +1 row written")
    print(f"  alpha_experiments: {len(alpha_rows)} rows written")


if __name__ == "__main__":
    experiment_id = sys.argv[1] if len(sys.argv) > 1 else settings.EXPERIMENT_ID
    asyncio.run(main(experiment_id))
