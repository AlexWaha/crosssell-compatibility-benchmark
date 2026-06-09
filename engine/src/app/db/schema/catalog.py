"""Clean project catalog schema (database avtc_catalog).

No table prefixes, no OpenCart cruft, only fields the app uses. Slugs live directly on
products/categories (no separate seo_urls table). Populated exclusively from Dataset/json/*.
product_ai_data holds LLM-normalized attributes + embedding text produced by the engine.

Engine owns this DDL (ADR-003). api and worker are read-only consumers.
"""

TABLES: list[str] = [
    "products",
    "categories",
    "product_categories",
    "product_attributes",
    "product_ai_data",
]

DDL: dict[str, str] = {
    "products": """
        CREATE TABLE IF NOT EXISTS `products` (
            `product_id`   INT UNSIGNED NOT NULL PRIMARY KEY,
            `slug`         VARCHAR(255) NOT NULL,
            `name`         VARCHAR(512) NOT NULL,
            `description`  TEXT NULL,
            `price`        DECIMAL(15,4) NOT NULL DEFAULT 0,
            `brand`        VARCHAR(128) NULL,
            `product_type` VARCHAR(128) NULL,
            `image_path`   VARCHAR(512) NULL,
            `status`       TINYINT(1) NOT NULL DEFAULT 1,
            `date_added`   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY `uq_slug` (`slug`),
            INDEX `idx_status` (`status`),
            INDEX `idx_brand` (`brand`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "categories": """
        CREATE TABLE IF NOT EXISTS `categories` (
            `category_id` INT UNSIGNED NOT NULL PRIMARY KEY,
            `parent_id`   INT UNSIGNED NOT NULL DEFAULT 0,
            `name`        VARCHAR(255) NOT NULL,
            `slug`        VARCHAR(255) NOT NULL,
            `sort_order`  INT NOT NULL DEFAULT 0,
            `status`      TINYINT(1) NOT NULL DEFAULT 1,
            UNIQUE KEY `uq_slug` (`slug`),
            INDEX `idx_parent` (`parent_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "product_categories": """
        CREATE TABLE IF NOT EXISTS `product_categories` (
            `product_id`  INT UNSIGNED NOT NULL,
            `category_id` INT UNSIGNED NOT NULL,
            PRIMARY KEY (`product_id`, `category_id`),
            INDEX `idx_category` (`category_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "product_attributes": """
        CREATE TABLE IF NOT EXISTS `product_attributes` (
            `product_id`      INT UNSIGNED NOT NULL,
            `attribute_name`  VARCHAR(191) NOT NULL,
            `attribute_value` TEXT NULL,
            PRIMARY KEY (`product_id`, `attribute_name`),
            INDEX `idx_product` (`product_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "product_ai_data": """
        CREATE TABLE IF NOT EXISTS `product_ai_data` (
            `product_id`         INT UNSIGNED NOT NULL PRIMARY KEY,
            `normalized_json`    JSON NULL,
            `compatibility_tags` JSON NULL,
            `product_type`       VARCHAR(128) NULL,
            `embedding_text`     TEXT NULL,
            `model_used`         VARCHAR(80) NULL,
            `source_hash`        CHAR(64) NULL,
            `processed_at`       TIMESTAMP NULL,
            `version`            INT UNSIGNED NOT NULL DEFAULT 1
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}


def get_create_sql(table: str) -> str:
    return DDL[table]
