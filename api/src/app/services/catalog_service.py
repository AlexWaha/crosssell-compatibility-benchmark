"""Catalog service: slug/id resolution, card assembly, recommendation join.

Thin domain layer over CatalogRepository and MetricsReadRepository.
All business logic that was inline in main.py handlers lives here.
"""

from __future__ import annotations

import logging

from app.db.catalog_repository import CatalogRepository
from app.db.metrics_repository import MetricsReadRepository

log = logging.getLogger(__name__)


class CatalogService:
    """Orchestrates catalog and metrics reads for product-related endpoints.

    Args:
        catalog: Connected CatalogRepository.
        metrics: Connected MetricsReadRepository.
    """

    def __init__(
        self, catalog: CatalogRepository, metrics: MetricsReadRepository
    ) -> None:
        self._catalog = catalog
        self._metrics = metrics

    async def resolve_id(self, ident: str, prefix: str) -> int | None:
        """Resolve a string identifier to an integer ID.

        If the string is all digits, returns it as an int directly.
        Otherwise performs a slug lookup in the catalog.

        Args:
            ident: Raw identifier string (numeric id or slug).
            prefix: "product_id" or "category_id" to select the right table.

        Returns:
            Integer ID, or None if not found.
        """
        if ident.isdigit():
            return int(ident)
        return await self._catalog.id_by_slug(prefix, ident)

    async def get_recommendations(
        self, product_id: int, limit: int, experiment: str
    ) -> dict:
        """Return recommendation items for a product, merging catalog cards with scores.

        Args:
            product_id: Source product ID.
            limit: Maximum recommendations to return.
            experiment: Experiment ID string.

        Returns:
            Dict with "product_id" and "items" (list of card+scores dicts).
        """
        rows = await self._metrics.recommendation_rows(product_id, limit, experiment)
        rec_ids = [r["recommended_id"] for r in rows]
        cards = await self._catalog.get_cards_by_ids(rec_ids)
        items = []
        for r in rows:
            card = cards.get(r["recommended_id"])
            if card:
                items.append(
                    {
                        **card,
                        "context_code": r["context_code"],
                        "hybrid_score": float(r["hybrid_score"]),
                        "semantic_score": float(r["semantic_score"]),
                        "logical_score": float(r["logical_score"]),
                    }
                )
        return {"product_id": product_id, "items": items}

    async def get_top_products(self, limit: int, experiment: str) -> dict:
        """Return top recommended products by recommendation count.

        Args:
            limit: Maximum number of products to return.
            experiment: Experiment ID string.

        Returns:
            Dict with "items" (list of card+reco_count dicts).
        """
        pairs = await self._metrics.top_recommended(limit, experiment)
        cards = await self._catalog.get_cards_by_ids([pid for pid, _ in pairs])
        items = [
            {**cards[pid], "reco_count": cnt} for pid, cnt in pairs if pid in cards
        ]
        return {"items": items}
