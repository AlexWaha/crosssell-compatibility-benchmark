"""Catalog DB repository (engine service - read and write helpers).

Provides read and write access to avtc_catalog for the catalog importer, normalizer,
and fan-out helpers. All SQL is parameterized. No DDL here - DDL lives in
app.db.schema.catalog.
"""

from __future__ import annotations

import json
import logging

from app.db.adapter import MySQLAdapter

log = logging.getLogger(__name__)


class CatalogRepository:
    """Catalog DB access for the engine service (read + write for product_ai_data).

    Args:
        db: Connected MySQLAdapter for the catalog database.
    """

    def __init__(self, db: MySQLAdapter) -> None:
        self._db = db

    async def active_product_ids(self) -> list[int]:
        """Return IDs of all active products in the catalog.

        Returns:
            List of product IDs where status = 1.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute("SELECT product_id FROM `products` WHERE status = 1")
            rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def load_raw_product(self, product_id: int) -> dict | None:
        """Load a product row, its category names, and its attributes as a dict.

        Args:
            product_id: Product to load.

        Returns:
            Dict with keys:
                product (dict): Row from products table.
                categories (list[str]): Category names the product belongs to.
                raw_attrs (dict[str, str]): attribute_name -> attribute_value mapping.
            Returns None if the product does not exist.
        """
        async with self._db.cursor(dict_cursor=True) as cur:
            await cur.execute(
                "SELECT product_id, name, description, brand, product_type, status "
                "FROM `products` WHERE product_id = %s",
                (product_id,),
            )
            product_row = await cur.fetchone()

        if product_row is None:
            return None

        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT c.name FROM `categories` c "
                "INNER JOIN `product_categories` pc ON pc.category_id = c.category_id "
                "WHERE pc.product_id = %s",
                (product_id,),
            )
            cat_rows = await cur.fetchall()

        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT attribute_name, attribute_value "
                "FROM `product_attributes` WHERE product_id = %s",
                (product_id,),
            )
            attr_rows = await cur.fetchall()

        categories = [row[0] for row in cat_rows]
        raw_attrs = {row[0]: row[1] for row in attr_rows if row[1] is not None}

        return {
            "product": dict(product_row),
            "categories": categories,
            "raw_attrs": raw_attrs,
        }

    async def iter_products_for_normalization(self, limit: int = 0) -> list[int]:
        """Return active product IDs eligible for normalization.

        Args:
            limit: Maximum number of IDs to return. 0 means all.

        Returns:
            List of product_id values (active products, ordered by product_id ASC).
        """
        sql = (
            "SELECT product_id FROM `products` WHERE status = 1 ORDER BY product_id ASC"
        )
        params: tuple = ()
        if limit > 0:
            sql += " LIMIT %s"
            params = (limit,)

        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(sql, params)
            rows = await cur.fetchall()
        return [row[0] for row in rows]

    async def get_ai_data_hash(self, product_id: int) -> str | None:
        """Return the stored source_hash for a product in product_ai_data.

        Args:
            product_id: Product to look up.

        Returns:
            source_hash string if the product has an ai_data row, None otherwise.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT source_hash FROM `product_ai_data` WHERE product_id = %s",
                (product_id,),
            )
            row = await cur.fetchone()
        return row[0] if row else None

    async def upsert_ai_data(self, row: dict) -> None:
        """Insert or update a product_ai_data row.

        Uses INSERT ... ON DUPLICATE KEY UPDATE so re-runs are idempotent.
        normalized_json and compatibility_tags are serialized to JSON strings.

        Args:
            row: Dict with keys: product_id, normalized_json, compatibility_tags,
                product_type, embedding_text, model_used, source_hash, version.
        """
        normalized_json = row.get("normalized_json")
        compatibility_tags = row.get("compatibility_tags")

        sql = (
            "INSERT INTO `product_ai_data` "
            "(`product_id`, `normalized_json`, `compatibility_tags`, `product_type`, "
            "`embedding_text`, `model_used`, `source_hash`, `processed_at`, `version`) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), %s) "
            "ON DUPLICATE KEY UPDATE "
            "`normalized_json` = VALUES(`normalized_json`), "
            "`compatibility_tags` = VALUES(`compatibility_tags`), "
            "`product_type` = VALUES(`product_type`), "
            "`embedding_text` = VALUES(`embedding_text`), "
            "`model_used` = VALUES(`model_used`), "
            "`source_hash` = VALUES(`source_hash`), "
            "`processed_at` = NOW(), "
            "`version` = VALUES(`version`)"
        )
        params = (
            row["product_id"],
            json.dumps(normalized_json, ensure_ascii=False)
            if normalized_json is not None
            else None,
            json.dumps(compatibility_tags, ensure_ascii=False)
            if compatibility_tags is not None
            else None,
            row.get("product_type"),
            row.get("embedding_text"),
            row.get("model_used"),
            row.get("source_hash"),
            row.get("version", 1),
        )
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(sql, params)
        log.debug("upserted product_ai_data product_id=%s", row["product_id"])

    async def done_normalized_ids(self) -> set[int]:
        """Return product_ids that already have a row in product_ai_data.

        Intended for Component B batch pre-filtering: callers can use this set to
        skip products that have already been normalized before issuing more expensive
        per-product queries.

        NOTE: This is NOT a substitute for hash-based idempotency.  A product present
        in this set may have been normalized with a different prompt version or model.
        The authoritative idempotency check is source_hash comparison via
        get_ai_data_hash(); this method is a coarse pre-filter only.

        Returns:
            Set of product_id values present in product_ai_data (any version/hash).
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute("SELECT product_id FROM `product_ai_data`")
            rows = await cur.fetchall()
        return {row[0] for row in rows}

    async def attribute_vocab_by_type(self, top_n: int = 25) -> dict[str, list[str]]:
        """Build a per-type attribute vocabulary from product_ai_data.

        For each product_type, counts how frequently each JSON key appears
        across all products of that type and returns the top-N most common
        keys in frequency-descending order. Keys that appear in many products
        of a type are the reliable signals for rule generation.

        The aggregation is done in Python (one SELECT per type is avoided by
        fetching all rows in a single query and grouping in memory).

        Args:
            top_n: Maximum number of keys to return per type. Default 25.

        Returns:
            Dict mapping product_type -> list of attribute key strings,
            ordered by descending frequency. Types with NULL normalized_json
            or no parseable keys are omitted.
        """
        async with self._db.cursor(dict_cursor=False) as cur:
            await cur.execute(
                "SELECT `product_type`, `normalized_json` "
                "FROM `product_ai_data` "
                "WHERE `product_type` IS NOT NULL AND `normalized_json` IS NOT NULL"
            )
            rows = await cur.fetchall()

        # Accumulate key -> count per type using a nested dict.
        type_key_counts: dict[str, dict[str, int]] = {}
        for product_type, normalized_json_str in rows:
            try:
                data = json.loads(normalized_json_str)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(data, dict):
                continue
            bucket = type_key_counts.setdefault(product_type, {})
            for key in data.keys():
                bucket[key] = bucket.get(key, 0) + 1

        vocab: dict[str, list[str]] = {}
        for product_type, key_counts in type_key_counts.items():
            sorted_keys = sorted(key_counts, key=lambda k: key_counts[k], reverse=True)
            vocab[product_type] = sorted_keys[:top_n]

        log.info(
            "attribute_vocab_by_type: built vocab for %d product types (top_n=%d)",
            len(vocab),
            top_n,
        )
        return vocab
