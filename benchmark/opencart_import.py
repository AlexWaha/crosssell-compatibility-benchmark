"""Generate OpenCart 3 import SQL from the curated dataset (dataset_v2).

Targets the live demo store schema (prefix `dm_`, language_id=1, store_id=0),
captured directly from demo_db - no guessing, no old dumps.

Produces `opencart_import.sql`:
  - wipes managed catalog tables (idempotent reimport)
  - inserts categories (+ description, path, to_store, seo)
  - inserts manufacturers (from product brands)
  - inserts attribute groups + attributes (from attributes.json)
  - inserts products (+ description, to_category, to_store, attributes, seo, image)

Product/category ids are PRESERVED from the dataset so ground_truth_pairs.json
maps 1:1 onto OpenCart product_id.

Image column points at `catalog/products/<NNNN>/<FILE>.jpg` (copied separately).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT_SQL = ROOT / "opencart_import.sql"

PREFIX = "dm_"
LANG = 1
STORE = 0
STOCK_STATUS_ID = 7   # In Stock (from demo_db)
TAX_CLASS_ID = 0      # None - always valid
QUANTITY = 100
DATE = "2020-01-01"
NOW = "2020-01-01 00:00:00"
IMG_BASE = "catalog/products"  # relative to OpenCart image/ dir


def esc(v) -> str:
    if v is None:
        return "''"
    s = str(v)
    s = s.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{s}'"


def main() -> None:
    products = json.loads((ROOT / "products.json").read_text(encoding="utf-8"))
    categories = json.loads((ROOT / "categories.json").read_text(encoding="utf-8"))
    attr_groups = json.loads((ROOT / "attributes.json").read_text(encoding="utf-8"))

    cid2cat = {c["id"]: c for c in categories}
    used_keywords: set[str] = set()

    def keyword(slug: str, uid: int) -> str:
        k = slug or f"item-{uid}"
        if k in used_keywords:
            k = f"{k}-{uid}"
        used_keywords.add(k)
        return k

    lines: list[str] = []
    a = lines.append
    a("-- AVTC curated dataset -> OpenCart 3 import (demo_db schema, prefix dm_)")
    a("SET NAMES utf8mb4;")
    a("SET FOREIGN_KEY_CHECKS=0;")
    a("START TRANSACTION;")
    a("")

    # --- wipe managed tables ---
    a("-- wipe managed catalog tables")
    for t in (
        "category", "category_description", "category_path", "category_to_store",
        "manufacturer", "manufacturer_description",
        "attribute_group", "attribute_group_description", "attribute", "attribute_description",
        "product", "product_description", "product_to_category", "product_to_store",
        "product_attribute", "product_image",
    ):
        a(f"DELETE FROM `{PREFIX}{t}`;")
    a(f"DELETE FROM `{PREFIX}seo_url` WHERE `query` LIKE 'product_id=%' OR `query` LIKE 'category_id=%';")
    a("")

    # --- categories ---
    a("-- categories")
    for c in categories:
        cid = c["id"]
        col = c.get("column", 1)
        top = c.get("top", 0)
        a(
            f"INSERT INTO `{PREFIX}category` "
            f"(`category_id`,`image`,`parent_id`,`top`,`column`,`sort_order`,`status`,`date_added`,`date_modified`,`noindex`) "
            f"VALUES ({cid},'',{c.get('parent_id',0)},{int(bool(top))},{col},0,1,{esc(NOW)},{esc(NOW)},1);"
        )
        a(
            f"INSERT INTO `{PREFIX}category_description` "
            f"(`category_id`,`language_id`,`name`,`description`,`meta_title`,`meta_description`,`meta_keyword`,`meta_h1`) "
            f"VALUES ({cid},{LANG},{esc(c['name'])},'',{esc(c['name'])},'','','');"
        )
        a(f"INSERT INTO `{PREFIX}category_to_store` (`category_id`,`store_id`) VALUES ({cid},{STORE});")
        # path: ancestors (root..self)
        chain: list[int] = []
        x = cid
        seen: set[int] = set()
        while x and x in cid2cat and x not in seen:
            seen.add(x)
            chain.append(x)
            x = cid2cat[x].get("parent_id", 0)
        chain.reverse()
        for level, pid in enumerate(chain):
            a(f"INSERT INTO `{PREFIX}category_path` (`category_id`,`path_id`,`level`) VALUES ({cid},{pid},{level});")
        kw = keyword(c.get("slug", ""), cid)
        a(
            f"INSERT INTO `{PREFIX}seo_url` (`store_id`,`language_id`,`query`,`keyword`) "
            f"VALUES ({STORE},{LANG},'category_id={cid}',{esc(kw)});"
        )
    a("")

    # --- manufacturers (from brands) ---
    a("-- manufacturers")
    brands: dict[str, int] = {}
    next_mid = 1
    for p in products:
        b = (p.get("brand") or "").strip()
        if b and b not in brands:
            brands[b] = next_mid
            next_mid += 1
    for b, mid in brands.items():
        a(
            f"INSERT INTO `{PREFIX}manufacturer` (`manufacturer_id`,`name`,`image`,`sort_order`,`noindex`) "
            f"VALUES ({mid},{esc(b[:64])},'',0,1);"
        )
        a(
            f"INSERT INTO `{PREFIX}manufacturer_description` "
            f"(`manufacturer_id`,`language_id`,`description`,`meta_description`,`meta_keyword`,`meta_title`,`meta_h1`) "
            f"VALUES ({mid},{LANG},'','','','','');"
        )
    a("")

    # --- attribute groups + attributes ---
    a("-- attribute groups + attributes")
    for g in attr_groups:
        gid = g["attribute_group_id"]
        a(f"INSERT INTO `{PREFIX}attribute_group` (`attribute_group_id`,`sort_order`) VALUES ({gid},0);")
        a(
            f"INSERT INTO `{PREFIX}attribute_group_description` (`attribute_group_id`,`language_id`,`name`) "
            f"VALUES ({gid},{LANG},{esc(g['name'][:64])});"
        )
        for attr in g.get("attributes", []):
            aid = attr["attribute_id"]
            a(f"INSERT INTO `{PREFIX}attribute` (`attribute_id`,`attribute_group_id`,`sort_order`) VALUES ({aid},{gid},0);")
            a(
                f"INSERT INTO `{PREFIX}attribute_description` (`attribute_id`,`language_id`,`name`) "
                f"VALUES ({aid},{LANG},{esc(attr['name'][:64])});"
            )
    a("")

    # --- products ---
    a("-- products")
    for p in products:
        pid = p["id"]
        name = p["name"]
        model = (p.get("slug") or name)[:64]
        brand = (p.get("brand") or "").strip()
        mid = brands.get(brand, 0)
        price = p.get("price", 0) or 0
        # image path: image_path = catalog/product/<NNNN>/<FILE> -> catalog/products/<NNNN>/<FILE>
        img = ""
        ip = p.get("image_path", "")
        if ip.startswith("catalog/product/"):
            img = IMG_BASE + "/" + ip[len("catalog/product/"):]
        elif ip:
            img = ip
        a(
            f"INSERT INTO `{PREFIX}product` "
            f"(`product_id`,`model`,`sku`,`upc`,`ean`,`jan`,`isbn`,`mpn`,`location`,`quantity`,`stock_status_id`,"
            f"`image`,`manufacturer_id`,`shipping`,`price`,`points`,`tax_class_id`,`date_available`,`weight`,"
            f"`weight_class_id`,`length`,`width`,`height`,`length_class_id`,`subtract`,`minimum`,`sort_order`,"
            f"`status`,`viewed`,`date_added`,`date_modified`,`noindex`) "
            f"VALUES ({pid},{esc(model)},'','','','','','','',{QUANTITY},{STOCK_STATUS_ID},"
            f"{esc(img)},{mid},1,{price:.4f},0,{TAX_CLASS_ID},{esc(DATE)},0,0,0,0,0,0,1,1,0,1,0,{esc(NOW)},{esc(NOW)},1);"
        )
        tags = ",".join(p.get("tags") or [])
        a(
            f"INSERT INTO `{PREFIX}product_description` "
            f"(`product_id`,`language_id`,`name`,`description`,`tag`,`meta_title`,`meta_description`,`meta_keyword`,`meta_h1`) "
            f"VALUES ({pid},{LANG},{esc(name)},{esc(p.get('description',''))},{esc(tags)},{esc(name)},'','','');"
        )
        a(f"INSERT INTO `{PREFIX}product_to_store` (`product_id`,`store_id`) VALUES ({pid},{STORE});")
        for i, cat in enumerate(p.get("category_ids", [])):
            if cat not in cid2cat:
                continue
            main = 1 if i == 0 else 0
            a(
                f"INSERT INTO `{PREFIX}product_to_category` (`product_id`,`category_id`,`main_category`) "
                f"VALUES ({pid},{cat},{main});"
            )
        # attributes from specs {group_id: {attr_id: value}}
        for _gid, attrs in (p.get("specs") or {}).items():
            for aid, val in attrs.items():
                if val in (None, "", "-"):
                    continue
                a(
                    f"INSERT INTO `{PREFIX}product_attribute` (`product_id`,`attribute_id`,`language_id`,`text`) "
                    f"VALUES ({pid},{int(aid)},{LANG},{esc(val)});"
                )
        kw = keyword(p.get("slug", ""), pid)
        a(
            f"INSERT INTO `{PREFIX}seo_url` (`store_id`,`language_id`,`query`,`keyword`) "
            f"VALUES ({STORE},{LANG},'product_id={pid}',{esc(kw)});"
        )

    a("")
    a("COMMIT;")
    a("SET FOREIGN_KEY_CHECKS=1;")

    OUT_SQL.write_text("\n".join(lines), encoding="utf-8")
    size = OUT_SQL.stat().st_size
    print(f"wrote {OUT_SQL.name}: {len(lines)} lines, {size/1024/1024:.1f} MB")
    print(f"categories={len(categories)} products={len(products)} manufacturers={len(brands)}")


if __name__ == "__main__":
    main()
