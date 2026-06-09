"""DDL definitions for avtc_metrics database tables.

Tables correspond to metrics-specification.md section F.
All formulas reference the AVTC article exclusively.

Engine owns this DDL (ADR-003). api uses a read-only MetricsReadRepository.
"""

TABLES: list[str] = [
    "pipeline_runs",
    "stage_metrics",
    "compatibility_evaluations",
    "quality_snapshots",
    "alpha_experiments",
    "recommendations",
    "ground_truth",
    "rule_cache",
]

_DDL: dict[str, str] = {
    "pipeline_runs": """
        CREATE TABLE IF NOT EXISTS `pipeline_runs` (
            `run_id`            BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `product_id`        INT UNSIGNED NOT NULL,
            `experiment_id`     VARCHAR(64) NOT NULL DEFAULT 'baseline_v1',
            `job_id`            BIGINT UNSIGNED NULL,
            `pipeline_version`  VARCHAR(20) NOT NULL DEFAULT '1.0.0',
            `status`            ENUM('running','completed','failed') NOT NULL DEFAULT 'running',
            `started_at`        TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            `finished_at`       TIMESTAMP(3) NULL,
            `total_duration_ms` INT UNSIGNED NULL,
            `error_message`     TEXT NULL,
            INDEX `idx_product`    (`product_id`),
            INDEX `idx_pr_exp`     (`experiment_id`),
            INDEX `idx_status`     (`status`),
            INDEX `idx_started`    (`started_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "stage_metrics": """
        CREATE TABLE IF NOT EXISTS `stage_metrics` (
            `id`              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `run_id`          BIGINT UNSIGNED NOT NULL,
            `stage`           ENUM(
                'ingestion','normalization','embedding',
                'indexing','compatibility','recommendation'
            ) NOT NULL,
            `duration_ms`     INT UNSIGNED NOT NULL,
            `items_processed` INT UNSIGNED NOT NULL DEFAULT 0,
            `errors_count`    INT UNSIGNED NOT NULL DEFAULT 0,
            `llm_calls`       INT UNSIGNED NOT NULL DEFAULT 0,
            `tokens_input`    INT UNSIGNED NOT NULL DEFAULT 0,
            `tokens_output`   INT UNSIGNED NOT NULL DEFAULT 0,
            `cost_usd`        DECIMAL(10,6) NOT NULL DEFAULT 0,
            `metadata`        JSON NULL,
            `created_at`      TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            INDEX `idx_run_stage` (`run_id`, `stage`),
            FOREIGN KEY (`run_id`) REFERENCES `pipeline_runs`(`run_id`) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "compatibility_evaluations": """
        CREATE TABLE IF NOT EXISTS `compatibility_evaluations` (
            `id`                  BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `run_id`              BIGINT UNSIGNED NOT NULL,
            `experiment_id`       VARCHAR(64) NOT NULL DEFAULT 'baseline_v1',
            `product_i`           INT UNSIGNED NOT NULL,
            `product_j`           INT UNSIGNED NOT NULL,
            `context_code`        VARCHAR(50) NULL,
            `semantic_score`      DECIMAL(8,6) NOT NULL,
            `logical_score`       DECIMAL(8,6) NOT NULL,
            `hybrid_score`        DECIMAL(8,6) NOT NULL,
            `alpha_used`          DECIMAL(4,3) NOT NULL,
            `rules_evaluated`     SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `rules_passed`        SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `rules_failed`        SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `rules_undefined`     SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `verdict`             TINYINT(1) NOT NULL,
            `evidence_claims`     SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `hallucinated_claims` SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `created_at`          TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            INDEX `idx_run`     (`run_id`),
            INDEX `idx_ce_exp`  (`experiment_id`),
            INDEX `idx_pair`    (`product_i`, `product_j`),
            INDEX `idx_verdict` (`verdict`),
            FOREIGN KEY (`run_id`) REFERENCES `pipeline_runs`(`run_id`) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "quality_snapshots": """
        CREATE TABLE IF NOT EXISTS `quality_snapshots` (
            `id`                   BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `experiment_id`        VARCHAR(50) NOT NULL,
            `snapshot_at`          TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            `sample_size`          INT UNSIGNED NOT NULL,
            `tvr`                  DOUBLE NULL,
            `precision_at_5`       DOUBLE NULL,
            `precision_at_10`      DOUBLE NULL,
            `precision_at_20`      DOUBLE NULL,
            `recall_at_5`          DOUBLE NULL,
            `recall_at_10`         DOUBLE NULL,
            `recall_at_20`         DOUBLE NULL,
            `f1_at_5`              DOUBLE NULL,
            `f1_at_10`             DOUBLE NULL,
            `f1_at_20`             DOUBLE NULL,
            `ndcg_at_5`            DOUBLE NULL,
            `ndcg_at_10`           DOUBLE NULL,
            `ndcg_at_20`           DOUBLE NULL,
            `map_at_5`             DOUBLE NULL,
            `map_at_10`            DOUBLE NULL,
            `map_at_20`            DOUBLE NULL,
            `mrr`                  DOUBLE NULL,
            `hallucination_rate`   DOUBLE NULL,
            `cohens_kappa`         DOUBLE NULL,
            `hsv`                  DOUBLE NULL,
            `jit_ontology_eff`     DOUBLE NULL,
            `sld`                  DOUBLE NULL,
            `auto_rate`            DOUBLE NULL,
            `alpha_optimal`        DOUBLE NULL,
            `avg_cost_per_product` DOUBLE NULL,
            `throughput_ppm`       DOUBLE NULL,
            `metadata`             JSON NULL,
            INDEX `idx_experiment` (`experiment_id`),
            INDEX `idx_snapshot`   (`snapshot_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "alpha_experiments": """
        CREATE TABLE IF NOT EXISTS `alpha_experiments` (
            `id`              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `experiment_id`   VARCHAR(50) NOT NULL,
            `alpha_value`     DOUBLE NOT NULL,
            `hsv`             DOUBLE NOT NULL,
            `precision_at_10` DOUBLE NULL,
            `recall_at_10`    DOUBLE NULL,
            `f1_at_10`        DOUBLE NULL,
            `ndcg_at_10`      DOUBLE NULL,
            `sample_size`     INT UNSIGNED NOT NULL,
            `ci_lower`        DOUBLE NULL,
            `ci_upper`        DOUBLE NULL,
            `created_at`      TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
            INDEX `idx_experiment_alpha` (`experiment_id`, `alpha_value`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "recommendations": """
        CREATE TABLE IF NOT EXISTS `recommendations` (
            `id`              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `experiment_id`   VARCHAR(50) NOT NULL DEFAULT 'baseline_v1',
            `product_id`      INT UNSIGNED NOT NULL,
            `recommended_id`  INT UNSIGNED NOT NULL,
            `context_code`    VARCHAR(50) NULL,
            `semantic_score`  DOUBLE NOT NULL,
            `logical_score`   DOUBLE NOT NULL,
            `hybrid_score`    DOUBLE NOT NULL,
            `alpha_used`      DECIMAL(4,3) NOT NULL,
            `verdict`         TINYINT(1) NOT NULL,
            `computed_at`     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY `uq_pair` (`experiment_id`, `product_id`, `recommended_id`, `context_code`),
            INDEX `idx_product` (`experiment_id`, `product_id`, `hybrid_score` DESC),
            INDEX `idx_recommended` (`recommended_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "ground_truth": """
        CREATE TABLE IF NOT EXISTS `ground_truth` (
            `id`           BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            `product_i`    INT UNSIGNED NOT NULL,
            `product_j`    INT UNSIGNED NOT NULL,
            `context_code` VARCHAR(50) NULL,
            `label`        TINYINT(1) NOT NULL,
            `source`       ENUM('llm','human') NOT NULL DEFAULT 'llm',
            `judge_model`  VARCHAR(80) NULL,
            `rationale`    TEXT NULL,
            `created_at`   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY `uq_label` (`product_i`, `product_j`, `source`),
            INDEX `idx_i` (`product_i`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    "rule_cache": """
        CREATE TABLE IF NOT EXISTS `rule_cache` (
            `type_a`        VARCHAR(128) NOT NULL,
            `type_b`        VARCHAR(128) NOT NULL,
            `rules_json`    JSON NOT NULL,
            `rule_count`    SMALLINT UNSIGNED NOT NULL DEFAULT 0,
            `generated_by`  VARCHAR(80) NOT NULL DEFAULT 'llm',
            `source_hash`   CHAR(64) NOT NULL DEFAULT '',
            `created_at`    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (`type_a`, `type_b`),
            INDEX `idx_created` (`created_at`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
}


def get_create_table_sql(table_name: str) -> str:
    """Return CREATE TABLE DDL for the given table name.

    Raises:
        KeyError: If table_name is not a known metrics table.
    """
    return _DDL[table_name]
