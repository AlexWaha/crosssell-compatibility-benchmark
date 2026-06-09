"""Eval orchestration: loads data, computes all metrics, builds DB rows.

K values used: K in {5, 10, 20}.
Justification: derived from the quality_snapshots DDL (precision_at_5/10/20,
recall_at_5/10/20, ndcg_at_5/10/20 columns) and confirmed by the
metrics_snapshot() SELECT in api/src/app/db/metrics_repository.py (lines 150-154)
and the snap[0..8] mapping in api/src/app/services/metrics_service.py (lines 88-108).

snap[] contract (from api metrics_service.py lines 88-108, confirmed against the
SELECT in api metrics_repository.py lines 150-154):
  snap[0]  = precision_at_5
  snap[1]  = precision_at_10
  snap[2]  = precision_at_20
  snap[3]  = recall_at_5
  snap[4]  = recall_at_10
  snap[5]  = recall_at_20
  snap[6]  = ndcg_at_5
  snap[7]  = ndcg_at_10
  snap[8]  = ndcg_at_20
  snap[9]  = alpha_optimal
  snap[10] = tvr
  snap[11] = hsv
  snap[12] = sld

Data source:
- ALL pair-level metrics (TVR, SLD, HSV, alpha_sweep, avg S/L/H) use
  compatibility_evaluations (all evaluated pairs: verdict 0 and 1).
  Previously the evaluator read only recommendations (verdict=1 subset),
  which produced meaningless TVR=1.0 and SLD=0.0.
- Ranked retrieval metrics (Precision/Recall/F1/NDCG/MAP/MRR) group pairs
  by product_i and rank by hybrid_score; ground_truth provides relevance labels.
- Coverage (products with >=1 recommendation) is reported by /api/summary from
  the recommendations table directly; the evaluator does not recompute it.

Ground-truth awareness:
- GT-dependent metrics (Precision/Recall/F1/NDCG/MAP/MRR/kappa) are skipped (None)
  when the ground_truth table is empty for the experiment.
- Label-free metrics (TVR, SLD, HSV, alpha_sweep) compute from compatibility_evaluations.
- run_eval succeeds with empty GT: label-free fields are real numbers, GT fields are null.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from app.services.eval.metrics import (
    alpha_sweep,
    cohen_kappa,
    f1_at_k,
    hsv,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    sld,
    tvr,
)

if TYPE_CHECKING:
    from app.db.metrics_repository import MetricsRepository

log = logging.getLogger(__name__)

# K values for all ranked metrics -- derived from quality_snapshots DDL columns.
EVAL_K_VALUES: tuple[int, ...] = (5, 10, 20)


def build_snapshot_row(
    experiment_id: str,
    pairs: list[dict],
    gt_map: dict[int, set[int]],
    tau_s: float,
    tau_l: float,
) -> tuple[dict, list[dict]]:
    """Compute all quality metrics and assemble the quality_snapshots row dict.

    Columns in the returned dict match the quality_snapshots INSERT parameter list
    exactly (see MetricsRepository.write_quality_snapshot).

    The alpha_sweep curve is computed ONCE here and returned alongside the snapshot
    row so that run_evaluation can reuse it without a second bootstrap pass.

    Label sources:
    - TVR, SLD, HSV, alpha_optimal: LLM verdict V + stored scores (label-free).
    - Precision/Recall/F1/NDCG/MAP/MRR/kappa: human ground-truth (None when absent).

    Args:
        experiment_id: Experiment identifier.
        pairs: ALL evaluated pairs from compatibility_evaluations for this experiment
               (list of dicts with product_id, recommended_id, semantic_score,
               logical_score, hybrid_score, verdict).  Includes both verdict=0 and
               verdict=1 rows.
        gt_map: Dict mapping source product_id -> set of relevant recommended_ids.
                Empty dict when ground_truth table has no rows.
        tau_s: Semantic threshold from settings (COMPAT_TAU_S).
        tau_l: Logical threshold from settings (COMPAT_TAU_L).

    Returns:
        Tuple of (snapshot_dict, alpha_curve) where snapshot_dict is suitable for
        passing to write_quality_snapshot() and alpha_curve is the list of per-alpha
        dicts from alpha_sweep (21 entries).
    """
    n = len(pairs)
    log.info(
        "building snapshot: experiment=%s evaluated_pairs=%d gt_sources=%d",
        experiment_id,
        n,
        len(gt_map),
    )

    # --- Label-free metrics (over ALL evaluated pairs) ---
    all_verdicts = [int(r["verdict"]) for r in pairs]
    all_semantic = [float(r["semantic_score"]) for r in pairs]
    all_logical = [float(r["logical_score"]) for r in pairs]
    all_hybrid = [float(r["hybrid_score"]) for r in pairs]

    _tvr = tvr(all_verdicts)
    _sld = sld(all_semantic, all_logical, tau_s, tau_l)
    _hsv = hsv(all_hybrid, all_verdicts)

    # Alpha sweep over all evaluated pairs (label-free HSV; GT metrics when gt_map non-empty)
    curve, alpha_star = alpha_sweep(
        pairs, tau_s, tau_l, k=10, gt_map=gt_map if gt_map else None
    )
    log.info("alpha_sweep done: alpha*=%.2f hsv=%.4f", alpha_star, _hsv)

    # --- GT-dependent metrics (None when no ground-truth) ---
    has_gt = bool(gt_map)

    # Group all evaluated pairs by product_id, sorted by hybrid_score DESC
    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in pairs:
        grouped[int(r["product_id"])].append(r)
    # Sort each group by hybrid_score descending to get the actual ranking
    for pid in grouped:
        grouped[pid].sort(key=lambda r: float(r["hybrid_score"]), reverse=True)

    def _mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    # Per-K metrics
    metrics_by_k: dict[int, dict[str, float | None]] = {}
    for k in EVAL_K_VALUES:
        if not has_gt:
            metrics_by_k[k] = {
                "precision": None,
                "recall": None,
                "f1": None,
                "ndcg": None,
                "map": None,
            }
            continue

        p_vals, r_vals, f_vals, n_vals, ap_vals = [], [], [], [], []
        for pid, group in grouped.items():
            relevant = gt_map.get(pid, set())
            if not relevant:
                continue
            ranked_ids = [int(r["recommended_id"]) for r in group]
            p_vals.append(precision_at_k(relevant, ranked_ids, k))
            r_vals.append(recall_at_k(relevant, ranked_ids, k))
            f_vals.append(f1_at_k(relevant, ranked_ids, k))
            n_vals.append(ndcg_at_k(relevant, ranked_ids, k))
            ap_vals.append(map_at_k(relevant, ranked_ids, k))

        metrics_by_k[k] = {
            "precision": _mean(p_vals),
            "recall": _mean(r_vals),
            "f1": _mean(f_vals),
            "ndcg": _mean(n_vals),
            "map": _mean(ap_vals),
        }

    # MRR (single value across all sources)
    _mrr: float | None = None
    if has_gt:
        query_pairs = []
        for pid, group in grouped.items():
            relevant = gt_map.get(pid, set())
            if relevant:
                ranked_ids = [int(r["recommended_id"]) for r in group]
                query_pairs.append((relevant, ranked_ids))
        _mrr = mrr(query_pairs) if query_pairs else None

    # Cohen's kappa -- requires paired (GT label, LLM verdict) for same (i,j) pairs
    _kappa: float | None = None
    if has_gt:
        y_true, y_pred = [], []
        for r in pairs:
            pid = int(r["product_id"])
            rid = int(r["recommended_id"])
            relevant = gt_map.get(pid, set())
            if relevant:  # source product has GT
                y_true.append(1 if rid in relevant else 0)
                y_pred.append(int(r["verdict"]))
        _kappa = cohen_kappa(y_true, y_pred) if y_true else None

    # Hallucination rate: not computable from compatibility_evaluations in this path
    # (hallucinated_claims is available in the table but not loaded by load_evaluated_pairs).
    # Set to None -- the quality_snapshots column is nullable.
    hallucination_rate: float | None = None

    row: dict = {
        "experiment_id": experiment_id,
        "sample_size": n,
        "tvr": _tvr,
        "precision_at_5": metrics_by_k[5]["precision"],
        "precision_at_10": metrics_by_k[10]["precision"],
        "precision_at_20": metrics_by_k[20]["precision"],
        "recall_at_5": metrics_by_k[5]["recall"],
        "recall_at_10": metrics_by_k[10]["recall"],
        "recall_at_20": metrics_by_k[20]["recall"],
        "f1_at_5": metrics_by_k[5]["f1"],
        "f1_at_10": metrics_by_k[10]["f1"],
        "f1_at_20": metrics_by_k[20]["f1"],
        "ndcg_at_5": metrics_by_k[5]["ndcg"],
        "ndcg_at_10": metrics_by_k[10]["ndcg"],
        "ndcg_at_20": metrics_by_k[20]["ndcg"],
        "map_at_5": metrics_by_k[5]["map"],
        "map_at_10": metrics_by_k[10]["map"],
        "map_at_20": metrics_by_k[20]["map"],
        "mrr": _mrr,
        "hallucination_rate": hallucination_rate,
        "cohens_kappa": _kappa,
        "hsv": _hsv,
        "jit_ontology_eff": None,  # reserved; not yet computed
        "sld": _sld,
        "auto_rate": None,  # reserved; not yet computed
        "alpha_optimal": alpha_star,
        "avg_cost_per_product": None,  # requires stage_metrics join; out of scope for eval
        "throughput_ppm": None,  # same
        "metadata": None,
    }
    return row, curve


def build_alpha_rows(
    experiment_id: str,
    curve: list[dict],
) -> list[dict]:
    """Convert alpha_sweep curve entries into alpha_experiments DB rows.

    Column mapping (from alpha_experiments DDL in schema/metrics.py):
      alpha_value, hsv, precision_at_10, recall_at_10, f1_at_10, ndcg_at_10,
      sample_size, ci_lower, ci_upper.

    The curve entries use 'precision_at_k' / 'recall_at_k' etc (k=10 in sweep).

    Args:
        experiment_id: Experiment identifier.
        curve: List of alpha entry dicts from alpha_sweep().

    Returns:
        List of dicts for write_alpha_experiments().
    """
    rows = []
    for entry in curve:
        rows.append(
            {
                "experiment_id": experiment_id,
                "alpha_value": entry["alpha_value"],
                "hsv": entry["hsv"],
                "precision_at_10": entry["precision_at_k"],
                "recall_at_10": entry["recall_at_k"],
                "f1_at_10": entry["f1_at_k"],
                "ndcg_at_10": entry["ndcg_at_k"],
                "sample_size": entry["sample_size"],
                "ci_lower": entry["ci_lower"],
                "ci_upper": entry["ci_upper"],
            }
        )
    return rows


async def run_evaluation(
    experiment_id: str,
    repo: "MetricsRepository",
    tau_s: float,
    tau_l: float,
) -> tuple[dict, list[dict]]:
    """Load data, compute metrics, write snapshot and alpha rows.

    Data source: compatibility_evaluations (ALL evaluated pairs, not just
    verdict=1 recommendations).  This yields correct TVR, SLD, HSV, and
    alpha* over the full pair population.

    Does NOT modify any existing recommendation data (read-mostly).
    Writes one row to quality_snapshots and 21 rows to alpha_experiments.

    Args:
        experiment_id: Experiment to evaluate.
        repo: Engine MetricsRepository (write-side, used for reads + writes).
        tau_s: Semantic threshold (COMPAT_TAU_S from settings).
        tau_l: Logical threshold (COMPAT_TAU_L from settings).

    Returns:
        Tuple of (snapshot_row, alpha_rows) as written to the DB.
    """
    log.info(
        "loading evaluated pairs from compatibility_evaluations: experiment=%s",
        experiment_id,
    )
    pairs = await repo.load_evaluated_pairs(experiment_id)
    log.info("loaded %d evaluated pairs (verdict 0+1)", len(pairs))

    gt_map = await repo.load_ground_truth(experiment_id)
    log.info("loaded ground_truth for %d source products", len(gt_map))

    # build_snapshot_row computes alpha_sweep once and returns the curve alongside
    # the snapshot dict -- no second bootstrap pass needed.
    snapshot_row, curve = build_snapshot_row(experiment_id, pairs, gt_map, tau_s, tau_l)
    alpha_rows = build_alpha_rows(experiment_id, curve)

    await repo.write_quality_snapshot(snapshot_row)
    log.info(
        "wrote quality_snapshot: experiment=%s sample=%d",
        experiment_id,
        snapshot_row["sample_size"],
    )

    await repo.write_alpha_experiments(experiment_id, alpha_rows)
    log.info("wrote %d alpha_experiments rows", len(alpha_rows))

    return snapshot_row, alpha_rows
