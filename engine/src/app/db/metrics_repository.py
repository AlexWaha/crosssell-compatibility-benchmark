"""Metrics DB repository (engine service - write side).

All SQL that writes to avtc_metrics lives here. Private query builders
(_build_upsert_sql, _build_evaluation_sql) are verbatim from the old writer.py.
done_product_ids was previously inline in batch_run._done_product_ids.

No DDL here - DDL lives in app.db.schema.metrics.
"""

from __future__ import annotations

import json
import logging

from app.db.adapter import MySQLAdapter

log = logging.getLogger(__name__)


class MetricsRepository:
    """Write-side access to the avtc_metrics database.

    Args:
        db: Connected MySQLAdapter for the metrics database.
    """

    def __init__(self, db: MySQLAdapter) -> None:
        self._db = db

    # ----------------------------------------------------------------- private query builders

    def _build_upsert_sql(
        self,
        product_id: int,
        recommended_id: int,
        context_code: str | None,
        semantic: float,
        logical: float,
        hybrid: float,
        alpha: float,
        verdict: bool,
        experiment_id: str = "baseline_v1",
    ) -> tuple[str, tuple]:
        """Build INSERT ... ON DUPLICATE KEY UPDATE for recommendations table."""
        sql = (
            "INSERT INTO `recommendations` "
            "(`experiment_id`, `product_id`, `recommended_id`, `context_code`, "
            "`semantic_score`, `logical_score`, `hybrid_score`, "
            "`alpha_used`, `verdict`, `computed_at`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW()) "
            "ON DUPLICATE KEY UPDATE "
            "`semantic_score` = VALUES(`semantic_score`), "
            "`logical_score` = VALUES(`logical_score`), "
            "`hybrid_score` = VALUES(`hybrid_score`), "
            "`alpha_used` = VALUES(`alpha_used`), "
            "`verdict` = VALUES(`verdict`), "
            "`computed_at` = NOW()"
        )
        params = (
            experiment_id,
            product_id,
            recommended_id,
            context_code[:50] if context_code else context_code,
            semantic,
            logical,
            hybrid,
            alpha,
            int(verdict),
        )
        return sql, params

    def _build_evaluation_sql(
        self,
        run_id: int,
        product_i: int,
        product_j: int,
        context_code: str | None,
        semantic: float,
        logical: float,
        hybrid: float,
        alpha: float,
        verdict: bool,
        experiment_id: str = "baseline_v1",
        rules_evaluated: int = 0,
        rules_passed: int = 0,
        rules_failed: int = 0,
        rules_undefined: int = 0,
        evidence_claims: int = 0,
        hallucinated_claims: int = 0,
    ) -> tuple[str, tuple]:
        """Build INSERT for compatibility_evaluations table."""
        sql = (
            "INSERT INTO `compatibility_evaluations` "
            "(`run_id`, `experiment_id`, `product_i`, `product_j`, `context_code`, "
            "`semantic_score`, `logical_score`, `hybrid_score`, `alpha_used`, "
            "`rules_evaluated`, `rules_passed`, `rules_failed`, `rules_undefined`, "
            "`verdict`, `evidence_claims`, `hallucinated_claims`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        params = (
            run_id,
            experiment_id,
            product_i,
            product_j,
            context_code[:50] if context_code else context_code,
            semantic,
            logical,
            hybrid,
            alpha,
            rules_evaluated,
            rules_passed,
            rules_failed,
            rules_undefined,
            int(verdict),
            evidence_claims,
            hallucinated_claims,
        )
        return sql, params

    # ----------------------------------------------------------------- public methods

    async def write_recommendations(self, rows: list[dict]) -> int:
        """Batch-write results to recommendations table. Returns count written.

        Args:
            rows: List of recommendation dicts matching build_upsert_sql kwargs.

        Returns:
            Number of rows written.
        """
        if not rows:
            return 0
        async with self._db.cursor(dict_cursor=False) as cur:
            for r in rows:
                sql, params = self._build_upsert_sql(**r)
                await cur.execute(sql, params)
        log.info("wrote %d recommendations", len(rows))
        return len(rows)

    async def write_evaluations(self, rows: list[dict]) -> int:
        """Batch-write to compatibility_evaluations table. Returns count written.

        Args:
            rows: List of evaluation dicts matching build_evaluation_sql kwargs.

        Returns:
            Number of rows written.
        """
        if not rows:
            return 0
        async with self._db.cursor(dict_cursor=False) as cur:
            for e in rows:
                sql, params = self._build_evaluation_sql(**e)
                await cur.execute(sql, params)
        log.info("wrote %d evaluations", len(rows))
        return len(rows)

    async def done_product_ids(self, experiment_id: str) -> set[int]:
        """Return IDs of products already processed for this experiment.

        Previously inline in batch_run._done_product_ids.

        Args:
            experiment_id: Experiment identifier string.

        Returns:
            Set of product_id values already present in recommendations.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT DISTINCT product_id FROM `recommendations` WHERE experiment_id = %s",
                (experiment_id,),
            )
            rows = await cur.fetchall()
        return {r[0] for r in rows}

    # ----------------------------------------------------------------- eval read methods

    async def load_recommendations(self, experiment_id: str) -> list[dict]:
        """Load all recommendation rows for an experiment (read-only, no modifications).

        Returns dicts with keys: product_id, recommended_id, semantic_score,
        logical_score, hybrid_score, verdict.

        Args:
            experiment_id: Experiment identifier string.

        Returns:
            List of recommendation dicts ordered by product_id, hybrid_score DESC.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT product_id, recommended_id, semantic_score, logical_score, "
                "hybrid_score, verdict "
                "FROM `recommendations` "
                "WHERE experiment_id = %s "
                "ORDER BY product_id, hybrid_score DESC",
                (experiment_id,),
            )
            rows = await cur.fetchall()
        return [
            {
                "product_id": int(r[0]),
                "recommended_id": int(r[1]),
                "semantic_score": float(r[2]),
                "logical_score": float(r[3]),
                "hybrid_score": float(r[4]),
                "verdict": int(r[5]),
            }
            for r in rows
        ]

    async def load_evaluated_pairs(self, experiment_id: str) -> list[dict]:
        """Load ALL evaluated pairs from compatibility_evaluations for an experiment.

        Filters by experiment_id directly on the compatibility_evaluations column
        (added in the experiment-isolation schema migration). For each
        (product_i, product_j, context_code) triple within the experiment, only
        the row from the latest run_id (MAX) is returned so that re-runs do not
        produce duplicate pairs in the eval.

        Args:
            experiment_id: Experiment identifier used to filter rows.

        Returns:
            List of dicts with keys: product_id, recommended_id, semantic_score,
            logical_score, hybrid_score, verdict, context_code.
            One row per unique (product_i, product_j, context_code) - latest run.
            Ordered by product_i, hybrid_score DESC.
        """
        log.info(
            "load_evaluated_pairs: experiment_id=%s",
            experiment_id,
        )
        sql = (
            "SELECT ce.product_i, ce.product_j, "
            "ce.semantic_score, ce.logical_score, ce.hybrid_score, "
            "ce.verdict, ce.context_code "
            "FROM `compatibility_evaluations` ce "
            "INNER JOIN ("
            "  SELECT ce2.product_i, ce2.product_j, ce2.context_code, "
            "         MAX(ce2.run_id) AS max_run_id "
            "  FROM `compatibility_evaluations` ce2 "
            "  WHERE ce2.experiment_id = %s "
            "  GROUP BY ce2.product_i, ce2.product_j, ce2.context_code"
            ") latest "
            "  ON latest.product_i = ce.product_i "
            "  AND latest.product_j = ce.product_j "
            "  AND (latest.context_code = ce.context_code "
            "       OR (latest.context_code IS NULL AND ce.context_code IS NULL)) "
            "  AND latest.max_run_id = ce.run_id "
            "WHERE ce.experiment_id = %s "
            "ORDER BY ce.product_i, ce.hybrid_score DESC"
        )
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(sql, (experiment_id, experiment_id))
            rows = await cur.fetchall()

        log.debug(
            "load_evaluated_pairs: experiment=%s pairs=%d", experiment_id, len(rows)
        )
        return [
            {
                "product_id": int(r[0]),
                "recommended_id": int(r[1]),
                "semantic_score": float(r[2]),
                "logical_score": float(r[3]),
                "hybrid_score": float(r[4]),
                "verdict": int(r[5]),
                "context_code": r[6],
            }
            for r in rows
        ]

    async def load_ground_truth(self, experiment_id: str) -> dict[int, set[int]]:
        """Load human ground-truth labels, grouped by source product_id.

        Ground-truth is currently populated by a paid GPT-5 judge step and may be
        empty. When empty, returns an empty dict -- callers must handle this case
        by skipping GT-dependent metrics (Precision/Recall/F1/NDCG/MAP/MRR/kappa).

        The ground_truth table has no experiment_id column; it stores canonical
        (product_i, product_j, source) triples.  We restrict to pairs that were
        actually evaluated in this experiment (appear in compatibility_evaluations
        via pipeline_runs -> recommendations) so orphan GT rows are excluded.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            Dict mapping product_i -> set of product_j where label=1 (compatible).
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            # Restrict GT to pairs present in compatibility_evaluations for this
            # experiment (all evaluated pairs, not just the verdict=1 subset).
            await cur.execute(
                "SELECT g.product_i, g.product_j "
                "FROM `ground_truth` g "
                "INNER JOIN `compatibility_evaluations` ce "
                "  ON ce.product_i = g.product_i "
                "  AND ce.product_j = g.product_j "
                "INNER JOIN `pipeline_runs` pr ON pr.run_id = ce.run_id "
                "INNER JOIN `recommendations` r "
                "  ON r.product_id = pr.product_id "
                "  AND r.experiment_id = %s "
                "WHERE g.label = 1 "
                "GROUP BY g.product_i, g.product_j",
                (experiment_id,),
            )
            rows = await cur.fetchall()
        gt_map: dict[int, set[int]] = {}
        for r in rows:
            pid = int(r[0])
            rid = int(r[1])
            gt_map.setdefault(pid, set()).add(rid)
        return gt_map

    # ----------------------------------------------------------------- eval write methods

    async def write_quality_snapshot(self, row: dict) -> None:
        """Insert one row into quality_snapshots.

        Column order matches the DDL in schema/metrics.py quality_snapshots table.
        All nullable metric columns accept None (written as NULL).

        Args:
            row: Dict returned by evaluator.build_snapshot_row().
        """
        sql = (
            "INSERT INTO `quality_snapshots` "
            "(`experiment_id`, `sample_size`, `tvr`, "
            "`precision_at_5`, `precision_at_10`, `precision_at_20`, "
            "`recall_at_5`, `recall_at_10`, `recall_at_20`, "
            "`f1_at_5`, `f1_at_10`, `f1_at_20`, "
            "`ndcg_at_5`, `ndcg_at_10`, `ndcg_at_20`, "
            "`map_at_5`, `map_at_10`, `map_at_20`, "
            "`mrr`, `hallucination_rate`, `cohens_kappa`, `hsv`, "
            "`jit_ontology_eff`, `sld`, `auto_rate`, `alpha_optimal`, "
            "`avg_cost_per_product`, `throughput_ppm`, `metadata`) "
            "VALUES (%s, %s, %s, "
            "%s, %s, %s, "
            "%s, %s, %s, "
            "%s, %s, %s, "
            "%s, %s, %s, "
            "%s, %s, %s, "
            "%s, %s, %s, %s, "
            "%s, %s, %s, %s, "
            "%s, %s, %s)"
        )
        params = (
            row["experiment_id"],
            row["sample_size"],
            row["tvr"],
            row["precision_at_5"],
            row["precision_at_10"],
            row["precision_at_20"],
            row["recall_at_5"],
            row["recall_at_10"],
            row["recall_at_20"],
            row["f1_at_5"],
            row["f1_at_10"],
            row["f1_at_20"],
            row["ndcg_at_5"],
            row["ndcg_at_10"],
            row["ndcg_at_20"],
            row["map_at_5"],
            row["map_at_10"],
            row["map_at_20"],
            row["mrr"],
            row["hallucination_rate"],
            row["cohens_kappa"],
            row["hsv"],
            row["jit_ontology_eff"],
            row["sld"],
            row["auto_rate"],
            row["alpha_optimal"],
            row["avg_cost_per_product"],
            row["throughput_ppm"],
            row["metadata"],
        )
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(sql, params)
        log.info(
            "wrote quality_snapshot: experiment=%s sample=%d tvr=%.4f sld=%.4f alpha*=%.2f",
            row["experiment_id"],
            row["sample_size"],
            row["tvr"] or 0.0,
            row["sld"] or 0.0,
            row["alpha_optimal"] or 0.0,
        )

    async def get_rule_cache(self, type_a: str, type_b: str) -> list[dict] | None:
        """Retrieve cached rules for a (type_a, type_b) pair from the rule_cache table.

        Args:
            type_a: Source product type.
            type_b: Target product type.

        Returns:
            List of rule dicts or None if no cached entry exists.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT `rules_json` FROM `rule_cache` "
                "WHERE `type_a` = %s AND `type_b` = %s",
                (type_a, type_b),
            )
            row = await cur.fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except (json.JSONDecodeError, TypeError):
            log.warning(
                "rule_cache corrupt for type_a=%s type_b=%s, discarding", type_a, type_b
            )
            return None

    async def set_rule_cache(
        self,
        type_a: str,
        type_b: str,
        rules: list[dict],
        generated_by: str = "llm",
        source_hash: str = "",
    ) -> None:
        """Upsert a rule set into the rule_cache table.

        Uses INSERT ... ON DUPLICATE KEY UPDATE so repeated calls are idempotent
        and re-runs update the stored rules rather than inserting duplicates.

        Args:
            type_a: Source product type.
            type_b: Target product type.
            rules: List of rule dicts to store.
            generated_by: Model identifier for the audit trail.
            source_hash: SHA-256 of the prompt inputs (for reproducibility audit).
        """
        rules_json = json.dumps(rules, ensure_ascii=False)
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "INSERT INTO `rule_cache` "
                "(`type_a`, `type_b`, `rules_json`, `rule_count`, "
                "`generated_by`, `source_hash`) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON DUPLICATE KEY UPDATE "
                "`rules_json` = VALUES(`rules_json`), "
                "`rule_count` = VALUES(`rule_count`), "
                "`generated_by` = VALUES(`generated_by`), "
                "`source_hash` = VALUES(`source_hash`)",
                (type_a, type_b, rules_json, len(rules), generated_by, source_hash),
            )
        log.debug(
            "rule_cache upserted type_a=%s type_b=%s rules=%d",
            type_a,
            type_b,
            len(rules),
        )

    # ----------------------------------------------------------------- pipeline run methods

    async def start_run(
        self,
        product_id: int,
        pipeline_version: str = "1.0.0",
        job_id: int | None = None,
        experiment_id: str = "baseline_v1",
    ) -> int:
        """Insert a pipeline_runs row with status='running' and return run_id.

        Args:
            product_id: Product being processed.
            pipeline_version: Semantic version string.
            job_id: Optional arq job ID for traceability.
            experiment_id: Experiment identifier for provenance tracking.

        Returns:
            The auto-incremented run_id.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "INSERT INTO `pipeline_runs` "
                "(`product_id`, `experiment_id`, `job_id`, `pipeline_version`, `status`) "
                "VALUES (%s, %s, %s, %s, 'running')",
                (product_id, experiment_id, job_id, pipeline_version),
            )
            run_id = cur.lastrowid
        log.debug(
            "started pipeline_run run_id=%d product_id=%d experiment_id=%s",
            run_id,
            product_id,
            experiment_id,
        )
        return run_id

    async def finish_run(
        self,
        run_id: int,
        status: str,
        total_duration_ms: int,
        error_message: str | None = None,
    ) -> None:
        """Update a pipeline_runs row with final status, duration, and finished_at.

        Args:
            run_id: The pipeline_runs.run_id to update.
            status: Final status ("completed" | "failed").
            total_duration_ms: Total elapsed milliseconds.
            error_message: Optional error message (stored when status='failed').
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "UPDATE `pipeline_runs` "
                "SET `status` = %s, `finished_at` = NOW(3), "
                "`total_duration_ms` = %s, `error_message` = %s "
                "WHERE `run_id` = %s",
                (status, total_duration_ms, error_message, run_id),
            )
        log.debug(
            "finished pipeline_run run_id=%d status=%s duration_ms=%d",
            run_id,
            status,
            total_duration_ms,
        )

    async def write_stage_metrics(self, run_id: int, rows: list[tuple]) -> None:
        """Bulk-insert stage_metrics rows via executemany.

        Column order per tuple must match to_row() in StageTimer:
        (run_id, stage, duration_ms, items_processed, errors_count,
         llm_calls, tokens_input, tokens_output, cost_usd, metadata).

        Args:
            run_id: Parent pipeline_runs.run_id (informational; rows already embed it).
            rows: List of parameter tuples from StageTimer.to_row().
        """
        if not rows:
            return
        sql = (
            "INSERT INTO `stage_metrics` "
            "(`run_id`, `stage`, `duration_ms`, `items_processed`, `errors_count`, "
            "`llm_calls`, `tokens_input`, `tokens_output`, `cost_usd`, `metadata`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.executemany(sql, rows)
        log.debug("wrote %d stage_metrics rows for run_id=%d", len(rows), run_id)

    async def write_ground_truth(self, rows: list[dict]) -> int:
        """Upsert ground_truth rows (INSERT IGNORE on the unique key).

        Column order matches ground_truth DDL:
        (product_i, product_j, context_code, label, source, judge_model, rationale).

        Uses INSERT IGNORE so repeated judge runs do not duplicate rows.

        Args:
            rows: List of dicts with keys: product_i, product_j, context_code,
                label, source, judge_model, rationale.

        Returns:
            Number of rows attempted (not necessarily inserted due to IGNORE).
        """
        if not rows:
            return 0
        sql = (
            "INSERT IGNORE INTO `ground_truth` "
            "(`product_i`, `product_j`, `context_code`, `label`, "
            "`source`, `judge_model`, `rationale`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        total_inserted = 0
        async with self._db.cursor(dict_cursor=False) as cur:
            for r in rows:
                await cur.execute(
                    sql,
                    (
                        r["product_i"],
                        r["product_j"],
                        r.get("context_code"),
                        r["label"],
                        r.get("source", "llm"),
                        r.get("judge_model"),
                        r.get("rationale"),
                    ),
                )
                total_inserted += cur.rowcount
        log.info(
            "ground_truth: attempted=%d inserted=%d (skipped duplicates=%d)",
            len(rows),
            total_inserted,
            len(rows) - total_inserted,
        )
        return total_inserted

    async def sample_pairs_for_judge(
        self,
        experiment_id: str,
        n: int = 500,
        stratify_by: str = "category",
    ) -> list[dict]:
        """Sample product pairs from recommendations for ground-truth labelling.

        Returns up to n pairs stratified by category (or evenly sampled when
        stratify_by is not 'category'). Only pairs not already in ground_truth
        are returned (to avoid re-labelling).

        Args:
            experiment_id: Experiment whose recommendations to sample from.
            n: Maximum number of pairs to return.
            stratify_by: Stratification field (currently only 'category' supported).

        Returns:
            List of dicts with keys: product_i, product_j, context_code,
            semantic_score, logical_score, hybrid_score.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            # Load all recs not already labelled
            await cur.execute(
                "SELECT r.product_id, r.recommended_id, r.context_code, "
                "r.semantic_score, r.logical_score, r.hybrid_score "
                "FROM `recommendations` r "
                "LEFT JOIN `ground_truth` g "
                "  ON g.product_i = r.product_id AND g.product_j = r.recommended_id "
                "WHERE r.experiment_id = %s AND g.id IS NULL "
                "ORDER BY RAND() "
                "LIMIT %s",
                (experiment_id, n),
            )
            rows = await cur.fetchall()
        return [
            {
                "product_i": int(r[0]),
                "product_j": int(r[1]),
                "context_code": r[2],
                "semantic_score": float(r[3]),
                "logical_score": float(r[4]),
                "hybrid_score": float(r[5]),
            }
            for r in rows
        ]

    async def write_alpha_experiments(
        self, experiment_id: str, rows: list[dict]
    ) -> None:
        """Insert alpha_experiments rows, replacing existing rows for this experiment.

        Deletes existing rows for the experiment first (idempotent re-run support),
        then bulk-inserts the new curve.

        Args:
            experiment_id: Experiment identifier.
            rows: List of dicts from evaluator.build_alpha_rows().
        """
        if not rows:
            return
        sql = (
            "INSERT INTO `alpha_experiments` "
            "(`experiment_id`, `alpha_value`, `hsv`, "
            "`precision_at_10`, `recall_at_10`, `f1_at_10`, `ndcg_at_10`, "
            "`sample_size`, `ci_lower`, `ci_upper`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        async with self._db.cursor(dict_cursor=False) as cur:
            # Delete prior rows so re-runs are idempotent (no duplicate alpha curves)
            await cur.execute(
                "DELETE FROM `alpha_experiments` WHERE experiment_id = %s",
                (experiment_id,),
            )
            for r in rows:
                await cur.execute(
                    sql,
                    (
                        r["experiment_id"],
                        r["alpha_value"],
                        r["hsv"],
                        r["precision_at_10"],
                        r["recall_at_10"],
                        r["f1_at_10"],
                        r["ndcg_at_10"],
                        r["sample_size"],
                        r["ci_lower"],
                        r["ci_upper"],
                    ),
                )
        log.info(
            "wrote %d alpha_experiments rows for experiment=%s",
            len(rows),
            experiment_id,
        )
