"""Read-only catalog repository for the api service.

All SQL that reads from avtc_catalog lives here. No DDL - engine owns all
schema creation (ADR-003).

Tables: products, categories, product_categories, product_attributes, product_ai_data.
"""

from __future__ import annotations

import contextlib
import json
import logging

from app.db.adapter import MySQLAdapter

log = logging.getLogger(__name__)

# product_type prefers the AI-normalized value, falling back to the catalog column.
_CARD_SELECT = (
    "p.product_id, p.slug, p.name, p.price, p.image_path, p.brand, "
    "COALESCE(ai.product_type, p.product_type) AS product_type "
)
_CARD_FROM = (
    "FROM `products` p LEFT JOIN `product_ai_data` ai ON ai.product_id = p.product_id "
)


def _card(row: dict) -> dict:
    """Build a product card dict from a DB row.

    Args:
        row: Dict row from aiomysql DictCursor.

    Returns:
        Product card dict with normalized fields.
    """
    return {
        "id": row["product_id"],
        "slug": row.get("slug") or "",
        "name": row.get("name") or "",
        "brand": row.get("brand") or "",
        "product_type": row.get("product_type") or "",
        "price": float(row["price"]) if row.get("price") is not None else None,
        "currency": "USD",
        "image": (row.get("image_path") or "").removeprefix("catalog/product/"),
    }


class CatalogRepository:
    """Read-only repository over the avtc_catalog database.

    Args:
        db: A connected MySQLAdapter instance.
    """

    def __init__(self, db: MySQLAdapter) -> None:
        self._db = db

    async def id_by_slug(self, prefix: str, slug: str) -> int | None:
        """Resolve a slug to an integer id.

        Args:
            prefix: "product_id" for products table, "category_id" for categories.
            slug: The slug string to look up.

        Returns:
            The integer id, or None if not found.
        """
        table, col = (
            ("products", "product_id")
            if prefix == "product_id"
            else ("categories", "category_id")
        )
        async with self._db.cursor() as cur:
            await cur.execute(
                f"SELECT `{col}` AS id FROM `{table}` WHERE slug = %s LIMIT 1", (slug,)
            )
            row = await cur.fetchone()
        return int(row["id"]) if row else None

    async def get_categories(self) -> list[dict]:
        """Return all active categories with product counts.

        Returns:
            List of category dicts ordered by parent_id, sort_order, category_id.
        """
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT c.category_id, c.parent_id, c.name, c.slug, "
                "COUNT(pc.product_id) AS product_count "
                "FROM `categories` c "
                "LEFT JOIN `product_categories` pc ON pc.category_id = c.category_id "
                "WHERE c.status = 1 "
                "GROUP BY c.category_id, c.parent_id, c.name, c.slug "
                "ORDER BY c.parent_id, c.sort_order, c.category_id"
            )
            rows = await cur.fetchall()
        return [
            {
                "id": r["category_id"],
                "parent_id": r["parent_id"],
                "name": r["name"],
                "slug": r["slug"] or str(r["category_id"]),
                "product_count": int(r["product_count"]),
            }
            for r in rows
        ]

    async def _subtree_ids(self, root_id: int) -> list[int]:
        """Return all category IDs in the subtree rooted at root_id.

        Args:
            root_id: The root category ID.

        Returns:
            List of all category IDs in the subtree (including root_id).
        """
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT category_id, parent_id FROM `categories` WHERE status = 1"
            )
            rows = await cur.fetchall()
        children: dict[int, list[int]] = {}
        for r in rows:
            children.setdefault(r["parent_id"], []).append(r["category_id"])
        out, stack = [root_id], [root_id]
        while stack:
            for c in children.get(stack.pop(), []):
                out.append(c)
                stack.append(c)
        return out

    async def get_category_products(
        self, category_id: int, page: int, page_size: int
    ) -> dict:
        """Return paginated products for a category subtree.

        Args:
            category_id: Root category ID.
            page: 1-based page number.
            page_size: Number of items per page.

        Returns:
            Dict with keys: total, page, page_size, items (list of product cards).
        """
        offset = (page - 1) * page_size
        ids = await self._subtree_ids(category_id)
        ph = ", ".join(["%s"] * len(ids))
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(DISTINCT p.product_id) AS total FROM `product_categories` pc "
                "JOIN `products` p ON p.product_id = pc.product_id AND p.status = 1 "
                f"WHERE pc.category_id IN ({ph})",
                tuple(ids),
            )
            total = (await cur.fetchone())["total"]
            await cur.execute(
                "SELECT "
                + _CARD_SELECT
                + _CARD_FROM
                + "JOIN `product_categories` pc ON pc.product_id = p.product_id "
                f"WHERE pc.category_id IN ({ph}) AND p.status = 1 "
                "GROUP BY p.product_id ORDER BY p.product_id LIMIT %s OFFSET %s",
                (*ids, page_size, offset),
            )
            rows = await cur.fetchall()
        return {
            "total": int(total),
            "page": page,
            "page_size": page_size,
            "items": [_card(r) for r in rows],
        }

    async def get_product(self, product_id: int) -> dict | None:
        """Return full product detail including attributes and category path.

        Args:
            product_id: The product ID to fetch.

        Returns:
            Product detail dict, or None if not found.
        """
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT p.product_id, p.slug, p.name, p.description, p.price, p.image_path, p.brand, "
                "COALESCE(ai.product_type, p.product_type) AS product_type, "
                "ai.normalized_json, ai.compatibility_tags "
                "FROM `products` p LEFT JOIN `product_ai_data` ai ON ai.product_id = p.product_id "
                "WHERE p.product_id = %s AND p.status = 1",
                (product_id,),
            )
            row = await cur.fetchone()
            if not row:
                return None
            await cur.execute(
                "SELECT c.category_id, c.name, c.slug FROM `product_categories` pc "
                "JOIN `categories` c ON c.category_id = pc.category_id "
                "WHERE pc.product_id = %s",
                (product_id,),
            )
            cats = await cur.fetchall()
            await cur.execute(
                "SELECT attribute_name, attribute_value FROM `product_attributes` WHERE product_id = %s",
                (product_id,),
            )
            attr_rows = await cur.fetchall()

        normalized: dict = {}
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            normalized = (
                json.loads(row["normalized_json"]) if row.get("normalized_json") else {}
            )
        tags: list = []
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            tags = (
                json.loads(row["compatibility_tags"])
                if row.get("compatibility_tags")
                else []
            )
        attributes = normalized.get("attributes") or {
            a["attribute_name"]: a["attribute_value"] for a in attr_rows
        }

        return {
            "id": row["product_id"],
            "slug": row.get("slug") or "",
            "name": row.get("name") or "",
            "brand": row.get("brand") or normalized.get("brand") or "",
            "product_type": row.get("product_type") or "",
            "price": float(row["price"]) if row.get("price") is not None else None,
            "currency": "USD",
            "image": (row.get("image_path") or "").removeprefix("catalog/product/"),
            "description": row.get("description") or "",
            "attributes": attributes,
            "compatibility_tags": tags,
            "category_path": [
                {"id": c["category_id"], "name": c["name"], "slug": c["slug"] or ""}
                for c in cats
            ],
        }

    async def get_cards_by_ids(self, ids: list[int]) -> dict[int, dict]:
        """Return product card dicts keyed by product_id.

        Args:
            ids: List of product IDs to fetch.

        Returns:
            Dict mapping product_id to card dict. Missing IDs are omitted.
        """
        if not ids:
            return {}
        ph = ", ".join(["%s"] * len(ids))
        async with self._db.cursor() as cur:
            await cur.execute(
                "SELECT "
                + _CARD_SELECT
                + _CARD_FROM
                + f"WHERE p.product_id IN ({ph}) AND p.status = 1",
                tuple(ids),
            )
            rows = await cur.fetchall()
        return {r["product_id"]: _card(r) for r in rows}

    async def catalog_counts(self) -> tuple[int, int, int]:
        """Return (products, categories, brands) active counts from the catalog.

        Returns:
            Tuple of (product_count, category_count, brand_count).
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute("SELECT COUNT(*) FROM `products` WHERE status = 1")
            products = (await cur.fetchone())[0]
            await cur.execute("SELECT COUNT(*) FROM `categories` WHERE status = 1")
            cats = (await cur.fetchone())[0]
            await cur.execute(
                "SELECT COUNT(DISTINCT brand) FROM `products` WHERE brand IS NOT NULL"
            )
            brands = (await cur.fetchone())[0]
        return int(products or 0), int(cats or 0), int(brands or 0)
