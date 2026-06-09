"""Unit tests for pure metric functions in app.services.eval.metrics.

Every test uses known-input / known-output so failures are diagnosable without
a running database. All metric functions are pure (no I/O).

Run: docker exec avtc_engine python -m pytest tests/ -q
"""

from __future__ import annotations

import math

from app.services.eval.metrics import (
    ALPHA_GRID,
    alpha_sweep,
    bootstrap_ci,
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

# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------


def test_precision_at_k_perfect():
    """All top-K items are relevant -> precision = 1.0."""
    assert precision_at_k({1, 2, 3}, [1, 2, 3, 4, 5], 3) == 1.0


def test_precision_at_k_none():
    """No relevant items in top-K -> precision = 0.0."""
    assert precision_at_k({10, 11}, [1, 2, 3], 3) == 0.0


def test_precision_at_k_partial():
    """1 of 3 top items is relevant -> precision = 1/3."""
    assert abs(precision_at_k({2}, [1, 2, 3], 3) - 1 / 3) < 1e-9


def test_precision_at_k_empty_ranked():
    assert precision_at_k({1}, [], 5) == 0.0


def test_precision_at_k_k_zero():
    assert precision_at_k({1}, [1, 2], 0) == 0.0


def test_precision_at_k_short_list():
    """Regression: ranked list shorter than k must use min(k, len(ranked)) as denominator.

    relevant={1,2,3}, ranked=[1,2,3] (3 items), k=5.
    All 3 are relevant; denominator = min(5,3) = 3.
    P@5 = 3/3 = 1.0.
    Old buggy code divided by k=5, returning 0.6.
    """
    assert precision_at_k({1, 2, 3}, [1, 2, 3], 5) == 1.0


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------


def test_recall_at_k_full():
    """All 3 relevant items are in top 5 -> recall = 1.0."""
    assert recall_at_k({1, 2, 3}, [1, 2, 3, 4, 5], 5) == 1.0


def test_recall_at_k_partial():
    """1 of 2 relevant items in top 3 -> recall = 0.5."""
    assert recall_at_k({1, 99}, [1, 2, 3], 3) == 0.5


def test_recall_at_k_empty_relevant():
    assert recall_at_k(set(), [1, 2, 3], 3) == 0.0


def test_recall_at_k_empty_ranked():
    assert recall_at_k({1}, [], 5) == 0.0


# ---------------------------------------------------------------------------
# f1_at_k
# ---------------------------------------------------------------------------


def test_f1_at_k_known():
    """Hand-verified: relevant={1,99}, ranked=[1,2,3], k=3.

    P@3 = 1/min(3,3) = 1/3  (1 hit in top-3; denominator=min(k,|ranked|)=3)
    R@3 = 1/2               (1 of 2 relevant items found)
    F1  = 2*(1/3)*(1/2) / ((1/3)+(1/2))
        = (1/3) / (5/6)
        = 2/5 = 0.4
    """
    assert abs(f1_at_k({1, 99}, [1, 2, 3], 3) - 0.4) < 1e-9


def test_f1_at_k_zero_when_no_overlap():
    assert f1_at_k({99}, [1, 2, 3], 3) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------


def test_ndcg_at_k_perfect_single():
    """Single relevant item at rank 1 -> NDCG = 1.0."""
    assert ndcg_at_k({1}, [1, 2, 3], 3) == 1.0


def test_ndcg_at_k_perfect_two_items():
    """Both relevant items in top 2 -> NDCG = 1.0 (ideal order)."""
    relevant = {1, 2}
    ranked = [1, 2, 3, 4]
    assert abs(ndcg_at_k(relevant, ranked, 4) - 1.0) < 1e-9


def test_ndcg_at_k_known():
    """Hand-worked example.

    relevant = {1, 3}; ranked = [2, 1, 3]; K=3
    DCG:
      pos1: item 2 not relevant -> (2^0 - 1)/log2(2) = 0
      pos2: item 1 relevant     -> (2^1 - 1)/log2(3) = 1/log2(3)
      pos3: item 3 relevant     -> (2^1 - 1)/log2(4) = 1/2
    IDCG (2 relevant in top 2):
      pos1: (2^1 - 1)/log2(2) = 1
      pos2: (2^1 - 1)/log2(3) = 1/log2(3)
    """
    relevant = {1, 3}
    ranked = [2, 1, 3]
    dcg = 0 + 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    result = ndcg_at_k(relevant, ranked, 3)
    assert abs(result - expected) < 1e-9


def test_ndcg_at_k_zero_when_no_relevant():
    assert ndcg_at_k(set(), [1, 2, 3], 3) == 0.0


def test_ndcg_at_k_no_hits():
    assert ndcg_at_k({99}, [1, 2, 3], 3) == 0.0


# ---------------------------------------------------------------------------
# map_at_k
# ---------------------------------------------------------------------------


def test_map_at_k_perfect():
    """relevant={1,2}, ranked=[1,2,3,4], K=4 -> AP = (1/1 + 2/2) / 2 = 1.0."""
    assert map_at_k({1, 2}, [1, 2, 3, 4], 4) == 1.0


def test_map_at_k_known():
    """relevant={1,3}, ranked=[2,1,3,4], K=4.
    hit at pos 2 (item 1): P = 1/2; hit at pos 3 (item 3): P = 2/3
    AP = (1/2 + 2/3) / 2 = (3/6 + 4/6) / 2 = 7/12.
    """
    result = map_at_k({1, 3}, [2, 1, 3, 4], 4)
    expected = (1 / 2 + 2 / 3) / 2
    assert abs(result - expected) < 1e-9


def test_map_at_k_zero_no_relevant():
    assert map_at_k(set(), [1, 2, 3], 3) == 0.0


def test_map_at_k_dense_relevant():
    """Regression: normalizer must be min(k, R), not |relevant|.

    relevant={1,2,3,4,5} (R=5), ranked=[1,2,3] all relevant, k=3.
    P@1=1/1, P@2=2/2=1, P@3=3/3=1; sum of P@i*rel_i = 1+1+1 = 3.
    Normalizer = min(3, 5) = 3.
    AP@3 = 3/3 = 1.0.
    Old buggy code divided by len(relevant)=5, returning 0.6.
    """
    assert map_at_k({1, 2, 3, 4, 5}, [1, 2, 3], 3) == 1.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------


def test_mrr_single_query_rank1():
    assert mrr([({1}, [1, 2, 3])]) == 1.0


def test_mrr_single_query_rank2():
    assert abs(mrr([({2}, [1, 2, 3])]) - 0.5) < 1e-9


def test_mrr_two_queries():
    """Q1: first hit at rank 1 (RR=1); Q2: first hit at rank 3 (RR=1/3). MRR=2/3."""
    queries = [({1}, [1, 2, 3]), ({3}, [1, 2, 3])]
    expected = (1.0 + 1 / 3) / 2
    assert abs(mrr(queries) - expected) < 1e-9


def test_mrr_no_hit():
    """No relevant item in ranked list -> RR=0 -> MRR=0."""
    assert mrr([({99}, [1, 2, 3])]) == 0.0


def test_mrr_empty():
    assert mrr([]) == 0.0


# ---------------------------------------------------------------------------
# tvr
# ---------------------------------------------------------------------------


def test_tvr_all_positive():
    assert tvr([1, 1, 1, 1]) == 1.0


def test_tvr_half():
    assert tvr([1, 0, 1, 0]) == 0.5


def test_tvr_empty():
    assert tvr([]) == 0.0


def test_tvr_known():
    """3 out of 5 are positive -> TVR = 0.6."""
    assert abs(tvr([1, 0, 1, 1, 0]) - 0.6) < 1e-9


# ---------------------------------------------------------------------------
# sld
# ---------------------------------------------------------------------------


def test_sld_no_discordance():
    """Both semantic and logical exceed their thresholds -> 0 discordant."""
    assert sld([0.8, 0.9], [0.7, 0.8], 0.3, 0.5) == 0.0


def test_sld_all_discordant():
    """S >= tau_S but L < tau_L for every pair -> SLD = 1.0."""
    semantic = [0.8, 0.9, 0.7]
    logical = [0.1, 0.2, 0.1]
    assert sld(semantic, logical, 0.3, 0.5) == 1.0


def test_sld_half_discordant():
    """2 of 4 pairs are discordant -> SLD = 0.5."""
    semantic = [0.8, 0.8, 0.1, 0.1]
    logical = [0.8, 0.1, 0.8, 0.1]
    # tau_s=0.5, tau_l=0.5
    # pair 0: (True, True) -> concordant
    # pair 1: (True, False) -> discordant
    # pair 2: (False, True) -> discordant
    # pair 3: (False, False) -> concordant
    assert sld(semantic, logical, 0.5, 0.5) == 0.5


def test_sld_empty():
    assert sld([], [], 0.3, 0.5) == 0.0


# ---------------------------------------------------------------------------
# hsv
# ---------------------------------------------------------------------------


def test_hsv_perfect_separation():
    """Scores perfectly separated by threshold 0.5: HSV = 1.0."""
    scores = [0.8, 0.9, 0.1, 0.2]
    verdicts = [1, 1, 0, 0]
    assert hsv(scores, verdicts) == 1.0


def test_hsv_random_agreement():
    """Half agree, half disagree -> HSV >= 0.5 (max over tau)."""
    scores = [0.8, 0.2]
    verdicts = [1, 0]
    result = hsv(scores, verdicts)
    assert result >= 0.5


def test_hsv_empty():
    assert hsv([], []) == 0.0


def test_hsv_explicit_tau():
    """With explicit single tau, agreement is deterministic."""
    scores = [0.6, 0.4]
    verdicts = [1, 0]
    # tau=0.5: score>=0.5 matches verdict=1 for item 0; score<0.5 matches verdict=0 for item 1
    result = hsv(scores, verdicts, tau_candidates=[0.5])
    assert result == 1.0


# ---------------------------------------------------------------------------
# alpha_sweep
# ---------------------------------------------------------------------------


def _make_recs(
    n: int = 10, semantic_base: float = 0.7, logical_base: float = 0.6
) -> list[dict]:
    """Generate n synthetic recommendation rows with alternating verdicts."""
    recs = []
    for i in range(n):
        recs.append(
            {
                "product_id": 1,
                "recommended_id": i + 100,
                "semantic_score": semantic_base - i * 0.02,
                "logical_score": logical_base - i * 0.02,
                "hybrid_score": 0.6 * (semantic_base - i * 0.02)
                + 0.4 * (logical_base - i * 0.02),
                "verdict": 1 if i < n // 2 else 0,
            }
        )
    return recs


def test_alpha_sweep_returns_21_points():
    """Sweep over {0.00, 0.05, ..., 1.00} = 21 alpha values."""
    recs = _make_recs(10)
    curve, _ = alpha_sweep(recs, tau_s=0.3, tau_l=0.5)
    assert len(curve) == 21


def test_alpha_sweep_alpha_star_in_grid():
    """alpha* must be one of the grid points."""
    recs = _make_recs(10)
    _, alpha_star = alpha_sweep(recs, tau_s=0.3, tau_l=0.5)
    assert alpha_star in ALPHA_GRID


def test_alpha_sweep_picks_best_hsv():
    """The alpha* entry must have the maximum HSV value in the curve."""
    recs = _make_recs(20)
    curve, alpha_star = alpha_sweep(recs, tau_s=0.3, tau_l=0.5)
    max_hsv = max(e["hsv"] for e in curve)
    alpha_star_hsv = next(e["hsv"] for e in curve if e["alpha_value"] == alpha_star)
    assert abs(alpha_star_hsv - max_hsv) < 1e-9


def test_alpha_sweep_empty_returns_default():
    """Empty rows -> empty curve, default alpha 0.6."""
    curve, alpha_star = alpha_sweep([], tau_s=0.3, tau_l=0.5)
    assert curve == []
    assert alpha_star == 0.6


def test_alpha_sweep_ci_bounds_valid():
    """CI lower <= HSV <= CI upper for every point."""
    recs = _make_recs(30)
    curve, _ = alpha_sweep(recs, tau_s=0.3, tau_l=0.5)
    for entry in curve:
        assert entry["ci_lower"] <= entry["hsv"] + 1e-9
        assert entry["ci_upper"] >= entry["hsv"] - 1e-9


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------


def test_kappa_perfect_agreement():
    """Identical arrays -> kappa = 1.0."""
    y = [1, 0, 1, 1, 0]
    assert abs(cohen_kappa(y, y) - 1.0) < 1e-9


def test_kappa_known_2x2():
    """Hand-computed 2x2:
    y_true = [1, 1, 0, 0], y_pred = [1, 0, 0, 1]
    tp=1 tn=1 fp=1 fn=1  n=4
    p_o = 2/4 = 0.5
    p_pos_true = 2/4 = 0.5; p_pos_pred = 2/4 = 0.5
    p_e = 0.5*0.5 + 0.5*0.5 = 0.5
    kappa = (0.5 - 0.5) / (1 - 0.5) = 0.0.
    """
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    result = cohen_kappa(y_true, y_pred)
    assert result is not None
    assert abs(result - 0.0) < 1e-9


def test_kappa_empty():
    assert cohen_kappa([], []) is None


def test_kappa_all_same_class_pred():
    """All predictions same class makes p_e=1 -> undefined (None)."""
    y_true = [1, 0, 1, 0]
    y_pred = [1, 1, 1, 1]
    # p_pos_pred = 1.0; p_neg_pred = 0.0; p_e = 0.5*1.0 + 0.5*0.0 = 0.5 -- not 1
    # This should NOT return None in this case
    result = cohen_kappa(y_true, y_pred)
    assert result is not None


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic():
    """Same seed -> same result on repeated calls."""
    values = [0.5, 0.6, 0.7, 0.4, 0.8]
    r1 = bootstrap_ci(values)
    r2 = bootstrap_ci(values)
    assert r1 == r2


def test_bootstrap_ci_ordering():
    """CI lower <= CI upper."""
    values = [0.3, 0.5, 0.7, 0.4, 0.6, 0.8]
    lo, hi = bootstrap_ci(values)
    assert lo <= hi


def test_bootstrap_ci_empty():
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_constant():
    """All same value -> CI is a single point."""
    lo, hi = bootstrap_ci([0.5] * 20)
    assert abs(lo - 0.5) < 1e-9
    assert abs(hi - 0.5) < 1e-9
