"""Metrics service: summary, metrics snapshot, and compare assembly.

Dict-building that was inline in main.py handlers now lives here.
Returns typed dicts that reproduce the exact current wire shape consumed by the SPA.
"""

from __future__ import annotations

import logging

from app.db.catalog_repository import CatalogRepository
from app.db.metrics_repository import MetricsReadRepository

log = logging.getLogger(__name__)


class MetricsService:
    """Assembles metrics response dicts for summary, metrics, and compare endpoints.

    Args:
        catalog: Connected CatalogRepository.
        metrics: Connected MetricsReadRepository.
    """

    def __init__(
        self, catalog: CatalogRepository, metrics: MetricsReadRepository
    ) -> None:
        self._catalog = catalog
        self._metrics = metrics

    async def summary(self, experiment_id: str) -> dict:
        """Build the /api/summary response dict.

        Args:
            experiment_id: Experiment to summarize.

        Returns:
            Dict matching the golden fixture for /api/summary.
        """
        catalog, _cats, _brands = await self._catalog.catalog_counts()
        data = await self._metrics.summary(experiment_id)
        total = data["total"]
        v1 = data["v1"]
        with_reco = data["with_reco"]
        return {
            "model": "gpt-5-nano",
            "catalog": catalog,
            "evaluated": data["evaluated"],
            "with_reco": with_reco,
            "coverage": (with_reco / catalog) if catalog else 0.0,
            "total_pairs": total,
            "verdict1_pairs": v1,
            "verdict0_pairs": total - v1,
            "verdict1_share": (v1 / total) if total else 0.0,
            "avg_semantic": float(data["asem"] or 0),
            "avg_logical": float(data["alog"] or 0),
            "avg_hybrid": float(data["ahyb"] or 0),
            "by_context": [
                {"code": c[0] or "other", "count": int(c[1])} for c in data["ctx"]
            ],
        }

    async def metrics_full(self, experiment_id: str) -> dict:
        """Build the /api/metrics response dict.

        Args:
            experiment_id: Experiment to query.

        Returns:
            Dict matching the golden fixture for /api/metrics.
        """
        products, cats, brands = await self._catalog.catalog_counts()
        data = await self._metrics.metrics_snapshot(experiment_id)
        with_reco = data["with_reco"]
        ctx_rows = data["ctx_rows"]
        snap = data["snap"]
        alpha_rows = data["alpha_rows"]

        ctx_total = sum(r[1] for r in ctx_rows) or 1
        context_dist = [
            {"code": r[0] or "other", "pct": r[1] / ctx_total} for r in ctx_rows
        ]

        def f(x: object) -> float:
            return float(x) if x is not None else 0.0

        if snap:
            p_at_k = [
                {
                    "k": 5,
                    "precision": f(snap[0]),
                    "recall": f(snap[3]),
                    "ndcg": f(snap[6]),
                },
                {
                    "k": 10,
                    "precision": f(snap[1]),
                    "recall": f(snap[4]),
                    "ndcg": f(snap[7]),
                },
                {
                    "k": 20,
                    "precision": f(snap[2]),
                    "recall": f(snap[5]),
                    "ndcg": f(snap[8]),
                },
            ]
            best_alpha = f(snap[9]) if snap[9] is not None else 0.6
            stats = {
                "index_build": "local",
                "tvr": f(snap[10]),
                "hsv": f(snap[11]),
                "sld": f(snap[12]),
            }
        else:
            p_at_k = [
                {"k": k, "precision": 0.0, "recall": 0.0, "ndcg": 0.0}
                for k in (5, 10, 20)
            ]
            best_alpha = 0.6
            stats = {"index_build": "local", "tvr": 0.0, "hsv": 0.0, "sld": 0.0}

        alpha = [{"alpha": f(r[0]), "quality": f(r[1])} for r in alpha_rows] or [
            {"alpha": 0.0, "quality": 0.0},
            {"alpha": 0.6, "quality": 0.0},
            {"alpha": 1.0, "quality": 0.0},
        ]
        coverage = (with_reco / products) if products else 0.0
        return {
            "catalog": {
                "products": products,
                "categories": cats,
                "with_reco": int(with_reco),
                "brands": brands,
            },
            "coverage": coverage,
            "contextDist": context_dist,
            "pAtK": p_at_k,
            "alpha": alpha,
            "best_alpha": best_alpha,
            "stats": stats,
        }

    async def compare(self, experiment_ids: list[str], catalog_count: int) -> dict:
        """Build the /api/compare response dict.

        Args:
            experiment_ids: List of experiment ID strings.
            catalog_count: Total active product count (pre-fetched).

        Returns:
            Dict matching the /api/compare wire shape.
        """
        rows = await self._metrics.compare(experiment_ids)
        out = []
        for r in rows:
            total = r["total"]
            v1 = r["v1"]
            with_reco = r["with_reco"]
            out.append(
                {
                    "experiment_id": r["experiment_id"],
                    "has_data": total > 0,
                    "coverage": (with_reco / catalog_count) if catalog_count else 0.0,
                    "with_reco": with_reco,
                    "total_pairs": total,
                    "verdict1_pairs": v1,
                    "verdict1_share": (v1 / total) if total else 0.0,
                    "avg_hybrid": float(r["ahyb"] or 0),
                    **r["snap"],
                }
            )
        return {"catalog": catalog_count, "experiments": out}
