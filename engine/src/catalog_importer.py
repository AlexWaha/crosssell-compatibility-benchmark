"""Build the clean avtc_catalog database exclusively from Dataset/json/*.

Idempotent: creates the database + tables, then upserts products, categories,
product_categories and product_attributes. The dataset is the single source of truth;
no OpenCart dependency. Attribute ids in products.specs are resolved to human names via
attributes.json.

Usage:
    python -m catalog_importer [--dataset /dataset/json]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os

import aiomysql

from app.core.config import settings
from app.db.schema.catalog import DDL, TABLES

log = logging.getLogger(__name__)


def _load(dataset_dir: str, name: str):
    with open(os.path.join(dataset_dir, name), encoding="utf-8") as f:
        return json.load(f)


def _attr_name_map(attributes: list) -> dict[int, str]:
    """attribute_id -> attribute name, from attributes.json groups."""
    out: dict[int, str] = {}
    for group in attributes:
        for a in group.get("attributes", []):
            out[int(a["attribute_id"])] = a["name"]
    return out


async def _ensure_db_and_tables(conn) -> None:
    async with conn.cursor() as cur:
        await cur.execute(
            "CREATE DATABASE IF NOT EXISTS `avtc_catalog` "
            "DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci"
        )
        await cur.execute("USE `avtc_catalog`")
        for t in TABLES:
            await cur.execute(DDL[t])


async def run(dataset_dir: str) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s"
    )
    products = _load(dataset_dir, "products.json")
    categories = _load(dataset_dir, "categories.json")
    attributes = _load(dataset_dir, "attributes.json")
    attr_names = _attr_name_map(attributes)
    log.info(
        "loaded %d products, %d categories, %d attribute defs",
        len(products),
        len(categories),
        len(attr_names),
    )

    conn = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        charset="utf8mb4",
        autocommit=False,
    )
    try:
        await _ensure_db_and_tables(conn)
        async with conn.cursor() as cur:
            await cur.execute("USE `avtc_catalog`")
            # clean rebuild (idempotent, dataset is source of truth)
            for t in (
                "product_attributes",
                "product_categories",
                "products",
                "categories",
            ):
                await cur.execute(f"TRUNCATE TABLE `{t}`")

            await cur.executemany(
                "INSERT INTO `categories` (category_id, parent_id, name, slug, sort_order, status) "
                "VALUES (%s,%s,%s,%s,%s,1)",
                [
                    (
                        c["id"],
                        c.get("parent_id", 0),
                        c["name"],
                        c["slug"],
                        c.get("column", 0),
                    )
                    for c in categories
                ],
            )

            prod_rows, pc_rows, attr_rows = [], [], []
            for p in products:
                pid = p["id"]
                prod_rows.append(
                    (
                        pid,
                        p["slug"],
                        p["name"],
                        p.get("description"),
                        p.get("price", 0),
                        p.get("brand"),
                        p.get("image_path"),
                    )
                )
                for cid in p.get("category_ids", []):
                    pc_rows.append((pid, cid))
                for _gid, specs in (p.get("specs") or {}).items():
                    for aid, val in specs.items():
                        nm = attr_names.get(int(aid))
                        if nm and val not in (None, ""):
                            attr_rows.append((pid, nm[:191], str(val)))

            await cur.executemany(
                "INSERT INTO `products` "
                "(product_id, slug, name, description, price, brand, image_path) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                prod_rows,
            )
            for i in range(0, len(pc_rows), 5000):
                await cur.executemany(
                    "INSERT IGNORE INTO `product_categories` "
                    "(product_id, category_id) VALUES (%s,%s)",
                    pc_rows[i : i + 5000],
                )
            for i in range(0, len(attr_rows), 5000):
                await cur.executemany(
                    "INSERT IGNORE INTO `product_attributes` "
                    "(product_id, attribute_name, attribute_value) VALUES (%s,%s,%s)",
                    attr_rows[i : i + 5000],
                )
        await conn.commit()
        log.info(
            "imported: products=%d product_categories=%d product_attributes=%d",
            len(prod_rows),
            len(pc_rows),
            len(attr_rows),
        )
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/dataset/json")
    asyncio.run(run(ap.parse_args().dataset))


if __name__ == "__main__":
    main()
