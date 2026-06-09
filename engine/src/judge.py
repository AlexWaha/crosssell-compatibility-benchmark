"""CLI: ground-truth judge for compatibility pairs.

Usage:
    python -m judge --experiment baseline_v1 --n 500 [--dry-run] [--authorize]

Default mode: MockJudge ($0, deterministic labels). Writes ground_truth rows.
Real mode: ONLY activated when BOTH --authorize flag is passed AND LLM_MODE=real.
           Uses JUDGE_MODEL (default: gpt-5) with strict JSON schema.

After ground_truth rows exist, run `python -m run_eval <experiment>` to get
non-null Precision/Recall/NDCG/MAP/MRR metrics.

Safety contract:
    - NEVER calls real OpenAI unless --authorize AND LLM_MODE=real simultaneously.
    - --dry-run: prints sample count only, writes ZERO rows.
    - Mock by default guards against accidental real LLM spend.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.adapter import MySQLAdapter
from app.db.metrics_repository import MetricsRepository
from app.services.compatibility.judge import MockJudge, judge_pair, select_sample
from app.services.llm.verifier import get_llm

log = logging.getLogger(__name__)


async def main(
    experiment_id: str,
    n: int,
    dry_run: bool,
    authorize: bool,
) -> None:
    """Run ground-truth judging for a sample of pairs in an experiment.

    Args:
        experiment_id: Experiment identifier (reads from recommendations table).
        n: Number of pairs to judge (stratified sample).
        dry_run: Print sample count only, write nothing.
        authorize: Enable real LLM judge (only when LLM_MODE=real).
    """
    setup_logging(settings.LOG_LEVEL)

    use_real = authorize and settings.LLM_MODE == "real"
    if authorize and settings.LLM_MODE != "real":
        print(
            "WARNING: --authorize requires LLM_MODE=real in environment. "
            "Falling back to mock judge."
        )
    if use_real:
        print(f"REAL judge mode: using model={settings.JUDGE_MODEL}")
    else:
        print("Mock judge mode ($0, deterministic labels)")

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
        pairs = await select_sample(repo, experiment_id, n=n)
        print(f"sample: {len(pairs)} pairs selected for experiment={experiment_id}")

        if dry_run:
            print(f"[dry-run] would judge {len(pairs)} pairs - no rows written")
            return

        if not pairs:
            print(
                "no unlabelled pairs found - ground_truth already populated or no recommendations"
            )
            return

        # Build judge (real or mock).
        if use_real:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.OPENAI_KEY.get_secret_value(),
                base_url=settings.OPENAI_BASE_URL or None,
            )
            llm = get_llm(settings, client)
            judge_model = settings.JUDGE_MODEL
        else:
            llm = None  # MockJudge does not use llm
            judge_model = "mock_judge"

        mock_judge = MockJudge(judge_model=judge_model) if not use_real else None

        gt_rows: list[dict] = []
        errors = 0

        for pair in pairs:
            product_i = pair["product_i"]
            product_j = pair["product_j"]
            source_doc = {
                "product_id": product_i,
                "name": f"product_{product_i}",
                "context_code": pair.get("context_code"),
            }
            cand_doc = {
                "product_id": product_j,
                "name": f"product_{product_j}",
            }

            try:
                if use_real:
                    row = await judge_pair(source_doc, cand_doc, llm, judge_model)
                else:
                    row = await mock_judge.judge_pair(source_doc, cand_doc)
                gt_rows.append(row)
            except Exception as exc:
                log.warning(
                    "judge failed for pair (%d, %d): %s", product_i, product_j, exc
                )
                errors += 1

        written = await repo.write_ground_truth(gt_rows)
        print(
            f"wrote {written} ground_truth rows "
            f"(errors={errors}, experiment={experiment_id})"
        )
        print(
            "run `python -m run_eval "
            + experiment_id
            + "` to compute Precision/Recall/NDCG"
        )

    finally:
        await metrics_adapter.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Ground-truth judge: label product pairs for quality evaluation."
    )
    ap.add_argument(
        "--experiment",
        default=settings.EXPERIMENT_ID,
        help="Experiment ID to sample pairs from (default: %(default)s)",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of pairs to judge (default: %(default)s)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sample count only, write no rows",
    )
    ap.add_argument(
        "--authorize",
        action="store_true",
        help="Enable real LLM judge (requires LLM_MODE=real in env)",
    )
    args = ap.parse_args()
    asyncio.run(
        main(
            experiment_id=args.experiment,
            n=args.n,
            dry_run=args.dry_run,
            authorize=args.authorize,
        )
    )
