"""Read-only catalog repository for the worker service.

The worker only needs one query: active product IDs for experiment fan-out.
No DDL - engine owns all schema creation (ADR-003).
"""

from __future__ import annotations

import logging

from app.db.adapter import MySQLAdapter

log = logging.getLogger(__name__)


class CatalogRepository:
    """Read-only repository over the avtc_catalog database.

    Args:
        db: A connected MySQLAdapter instance.
    """

    def __init__(self, db: MySQLAdapter) -> None:
        self._db = db

    async def active_product_ids(self) -> list[int]:
        """Return all active product IDs (status=1) from the products table.

        Returns:
            List of product_id integers for products with status=1.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT `product_id` FROM `products` WHERE `status` = %s", (1,)
            )
            rows = await cur.fetchall()
        ids = [row[0] for row in rows]
        log.info("active_product_ids: found %d active products", len(ids))
        return ids
