"""Read-only metrics repository for the api service.

All SQL that reads from avtc_metrics lives here. No writes, no DDL.
Engine owns all schema creation and write operations (ADR-003).

Tables: recommendations, quality_snapshots, alpha_experiments.
"""

from __future__ import annotations

import logging

from app.db.adapter import MySQLAdapter

log = logging.getLogger(__name__)

# Columns used in the compare endpoint quality_snapshots query
_SNAP_COLS = [
    "sample_size",
    "tvr",
    "precision_at_10",
    "recall_at_10",
    "f1_at_10",
    "ndcg_at_10",
    "map_at_10",
    "mrr",
    "hsv",
    "sld",
    "cohens_kappa",
    "alpha_optimal",
]


class MetricsReadRepository:
    """Read-only repository over the avtc_metrics database.

    Args:
        db: A connected MySQLAdapter instance.
    """

    def __init__(self, db: MySQLAdapter) -> None:
        self._db = db

    async def top_recommended(
        self, limit: int, experiment_id: str = "baseline_v1"
    ) -> list[tuple[int, int]]:
        """Return top recommended product IDs by recommendation count (verdict=1).

        Args:
            limit: Maximum number of results.
            experiment_id: Experiment to filter by.

        Returns:
            List of (product_id, count) tuples ordered by count DESC.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT product_id, COUNT(*) AS cnt FROM `recommendations` "
                "WHERE verdict = 1 AND experiment_id = %s "
                "GROUP BY product_id ORDER BY cnt DESC, product_id LIMIT %s",
                (experiment_id, limit),
            )
            return [(int(r[0]), int(r[1])) for r in await cur.fetchall()]

    async def recommendation_rows(
        self, product_id: int, limit: int, experiment_id: str = "baseline_v1"
    ) -> list[dict]:
        """Return recommendation rows for a product.

        Args:
            product_id: Source product ID.
            limit: Maximum number of results.
            experiment_id: Experiment to filter by.

        Returns:
            List of dicts with recommended_id, context_code, semantic_score,
            logical_score, hybrid_score.
        """
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT recommended_id, context_code, semantic_score, logical_score, hybrid_score "
                "FROM `recommendations` "
                "WHERE product_id = %s AND verdict = 1 AND experiment_id = %s "
                "ORDER BY hybrid_score DESC LIMIT %s",
                (product_id, experiment_id, limit),
            )
            return list(await cur.fetchall())

    async def summary(self, experiment_id: str) -> dict:
        """Return aggregate summary stats for an experiment.

        Args:
            experiment_id: Experiment to summarize.

        Returns:
            Dict with total, verdict counts, score averages, and context breakdown.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT COUNT(*), SUM(verdict = 1), COUNT(DISTINCT product_id), "
                "COUNT(DISTINCT CASE WHEN verdict = 1 THEN product_id END), "
                "AVG(semantic_score), AVG(logical_score), AVG(hybrid_score) "
                "FROM `recommendations` WHERE experiment_id = %s",
                (experiment_id,),
            )
            total, v1, evaluated, with_reco, asem, alog, ahyb = await cur.fetchone()
            await cur.execute(
                "SELECT context_code, COUNT(*) FROM `recommendations` "
                "WHERE verdict = 1 AND experiment_id = %s "
                "GROUP BY context_code ORDER BY 2 DESC LIMIT 10",
                (experiment_id,),
            )
            ctx = await cur.fetchall()
        return {
            "total": int(total or 0),
            "v1": int(v1 or 0),
            "evaluated": int(evaluated or 0),
            "with_reco": int(with_reco or 0),
            "asem": asem,
            "alog": alog,
            "ahyb": ahyb,
            "ctx": list(ctx),
        }

    async def metrics_snapshot(self, experiment_id: str) -> dict:
        """Return the latest quality snapshot for an experiment.

        Args:
            experiment_id: Experiment to query.

        Returns:
            Dict with with_reco count, context distribution rows, quality_snapshot
            tuple (or None), and alpha experiment rows.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT COUNT(DISTINCT product_id) FROM `recommendations` "
                "WHERE verdict = 1 AND experiment_id = %s",
                (experiment_id,),
            )
            with_reco = (await cur.fetchone())[0]
            await cur.execute(
                "SELECT context_code, COUNT(*) FROM `recommendations` "
                "WHERE verdict = 1 AND experiment_id = %s "
                "GROUP BY context_code ORDER BY 2 DESC",
                (experiment_id,),
            )
            ctx_rows = await cur.fetchall()
            await cur.execute(
                "SELECT precision_at_5, precision_at_10, precision_at_20, recall_at_5, recall_at_10, "
                "recall_at_20, ndcg_at_5, ndcg_at_10, ndcg_at_20, alpha_optimal, tvr, hsv, sld "
                "FROM `quality_snapshots` WHERE experiment_id = %s ORDER BY snapshot_at DESC LIMIT 1",
                (experiment_id,),
            )
            snap = await cur.fetchone()
            await cur.execute(
                "SELECT alpha_value, COALESCE(f1_at_10, hsv) FROM `alpha_experiments` "
                "WHERE experiment_id = %s ORDER BY alpha_value",
                (experiment_id,),
            )
            alpha_rows = await cur.fetchall()
        return {
            "with_reco": with_reco,
            "ctx_rows": list(ctx_rows),
            "snap": snap,
            "alpha_rows": list(alpha_rows),
        }

    async def compare(self, experiment_ids: list[str]) -> list[dict]:
        """Return compare rows for a list of experiment IDs.

        Args:
            experiment_ids: List of experiment ID strings.

        Returns:
            List of dicts, one per experiment, with coverage, pair counts, and
            quality snapshot columns.
        """
        rows = []
        async with self._db.cursor(dict_cursor=False) as cur:
            for exp in experiment_ids:
                await cur.execute(
                    "SELECT COUNT(*), SUM(verdict = 1), "
                    "COUNT(DISTINCT CASE WHEN verdict = 1 THEN product_id END), AVG(hybrid_score) "
                    "FROM `recommendations` WHERE experiment_id = %s",
                    (exp,),
                )
                total, v1, with_reco, ahyb = await cur.fetchone()
                total = int(total or 0)
                v1 = int(v1 or 0)
                with_reco = int(with_reco or 0)
                await cur.execute(
                    f"SELECT {', '.join(_SNAP_COLS)} FROM `quality_snapshots` "
                    "WHERE experiment_id = %s ORDER BY snapshot_at DESC LIMIT 1",
                    (exp,),
                )
                srow = await cur.fetchone()
                snap = (
                    {
                        c: (float(srow[i]) if srow[i] is not None else None)
                        for i, c in enumerate(_SNAP_COLS)
                    }
                    if srow
                    else {c: None for c in _SNAP_COLS}
                )
                rows.append(
                    {
                        "experiment_id": exp,
                        "total": total,
                        "v1": v1,
                        "with_reco": with_reco,
                        "ahyb": ahyb,
                        "snap": snap,
                    }
                )
        return rows
