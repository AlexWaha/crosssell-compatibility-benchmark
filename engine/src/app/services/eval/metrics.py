"""Pure metric functions for AVTC evaluation engine.

All functions are I/O-free and fully unit-testable. Formulas are taken verbatim
from docs/article2/methods.md (reconstructed from the article specification).

Label-source conventions (per methods.md):
- Precision/Recall/F1/NDCG/MAP/MRR/Cohen's kappa: require human ground-truth labels.
  When ground-truth is absent the caller passes an empty list and receives 0.0 / None.
- TVR, SLD: label-free; computed from stored verdict V (TINYINT 0/1) in recommendations.
- HSV / alpha_sweep: use verdict V as the proxy label when human GT is absent (the
  LLM-produced binary verdict is the signal being evaluated for consistency with the
  hybrid score).

Bootstrap uses a fixed seed (BOOTSTRAP_SEED = 42) for reproducibility. The seed is
documented here and in run_eval so results are deterministic across re-runs.

NOTE on RNG change (numpy vectorization):
    _bootstrap_ci_hsv previously used random.Random(BOOTSTRAP_SEED).choices() for
    resampling. It now uses numpy.random.default_rng(BOOTSTRAP_SEED).integers() for
    index-based resampling. This changes the exact resample draws (different RNG
    algorithm), so the raw CI bound values differ from the old pure-Python output.
    The statistic's statistical validity is unaffected: the seed is still fixed, the
    method is still percentile bootstrap with B=1000, and repeated calls with the
    same data produce identical results. HSV/TVR/SLD/alpha* point estimates are
    numerically identical to the pure-Python version (within float tolerance 1e-9).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

# Fixed seed for bootstrap CI -- must match run_eval documentation.
BOOTSTRAP_SEED = 42

# Alpha sweep grid (0.00, 0.05, ..., 1.00) -- 21 points per methods.md.
ALPHA_GRID: list[float] = [round(i * 0.05, 2) for i in range(21)]

# Default tau grid for HSV: 101 values 0.00..1.00
_TAU_DEFAULT: np.ndarray = np.linspace(0.0, 1.0, 101)


# ---------------------------------------------------------------------------
# Ranking metrics (require ground-truth relevance labels)
# ---------------------------------------------------------------------------


def precision_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """Precision@K: fraction of top-K retrieved items that are relevant.

    Formula (methods.md): P@K = |{i in top-K: rel(i)=1}| / min(K, |cand|)

    The denominator is min(K, len(ranked)) so that a short candidate list is not
    penalised for having fewer than K items (the system retrieved everything it
    could).

    Args:
        relevant: Set of relevant item IDs (ground-truth positives).
        ranked: Ranked list of retrieved item IDs (best first).
        k: Cutoff.

    Returns:
        Precision at K in [0.0, 1.0]. Returns 0.0 when K=0 or ranked is empty.
    """
    if k <= 0 or not ranked:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / min(k, len(ranked))


def recall_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """Recall@K: fraction of relevant items that appear in the top-K results.

    Formula (methods.md): R@K = |{i in top-K: rel(i)=1}| / |relevant|

    Args:
        relevant: Set of relevant item IDs.
        ranked: Ranked list of retrieved item IDs (best first).
        k: Cutoff.

    Returns:
        Recall at K in [0.0, 1.0]. Returns 0.0 when relevant is empty.
    """
    if not relevant or k <= 0 or not ranked:
        return 0.0
    top_k = ranked[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def f1_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """F1@K: harmonic mean of Precision@K and Recall@K.

    Formula (methods.md): F1@K = 2 * P@K * R@K / (P@K + R@K)

    Args:
        relevant: Set of relevant item IDs.
        ranked: Ranked list of retrieved item IDs (best first).
        k: Cutoff.

    Returns:
        F1 at K in [0.0, 1.0]. Returns 0.0 when both P and R are 0.
    """
    p = precision_at_k(relevant, ranked, k)
    r = recall_at_k(relevant, ranked, k)
    denom = p + r
    return 2.0 * p * r / denom if denom > 0.0 else 0.0


def ndcg_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """NDCG@K: normalised discounted cumulative gain.

    Formula (methods.md): DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i+1)
    NDCG@K = DCG@K / IDCG@K  where IDCG = DCG of ideal (all relevant first) ranking.

    Args:
        relevant: Set of relevant item IDs (binary relevance: 1 if in set, 0 otherwise).
        ranked: Ranked list of retrieved item IDs (best first).
        k: Cutoff.

    Returns:
        NDCG at K in [0.0, 1.0]. Returns 0.0 when relevant is empty or K=0.
    """
    if not relevant or k <= 0 or not ranked:
        return 0.0

    def _dcg(items: list[int], cutoff: int) -> float:
        gain = 0.0
        for i, item in enumerate(items[:cutoff], start=1):
            rel = 1.0 if item in relevant else 0.0
            gain += (2.0**rel - 1.0) / math.log2(i + 1)
        return gain

    dcg = _dcg(ranked, k)
    # Ideal ranking: all relevant items at the top, up to k
    ideal_count = min(len(relevant), k)
    idcg = sum((2.0**1.0 - 1.0) / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def map_at_k(relevant: set[int], ranked: list[int], k: int) -> float:
    """Average Precision@K (AP@K): area under the P-R curve up to cutoff K.

    Formula (methods.md): AP@K = (1/min(K, R)) * sum_{i=1}^{K} P@i * rel_i

    The normalizer is min(K, |relevant|) so that AP@K is bounded by 1.0 even
    when the candidate list is shorter than K.

    Args:
        relevant: Set of relevant item IDs.
        ranked: Ranked list of retrieved item IDs (best first).
        k: Cutoff.

    Returns:
        AP at K in [0.0, 1.0]. Returns 0.0 when relevant is empty.
    """
    if not relevant or k <= 0 or not ranked:
        return 0.0
    hits = 0
    score = 0.0
    for i, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(k, len(relevant))


def mrr(queries: list[tuple[set[int], list[int]]]) -> float:
    """Mean Reciprocal Rank across a list of (relevant, ranked) query pairs.

    Formula (methods.md): MRR = (1/|Q|) * sum_q 1/rank_q  where rank_q is the
    position of the first relevant item in ranked list for query q (0 if none).

    Args:
        queries: List of (relevant_set, ranked_list) tuples, one per query/source product.

    Returns:
        MRR in [0.0, 1.0]. Returns 0.0 when queries is empty.
    """
    if not queries:
        return 0.0
    total = 0.0
    for relevant, ranked in queries:
        for i, item in enumerate(ranked, start=1):
            if item in relevant:
                total += 1.0 / i
                break
    return total / len(queries)


# ---------------------------------------------------------------------------
# Label-free metrics (computed from stored V, S, L -- no human GT needed)
# ---------------------------------------------------------------------------


def tvr(verdicts: Sequence[int]) -> float:
    """True-Verdict Rate: fraction of evaluated pairs where LLM verdict V=1.

    Formula (methods.md): TVR = |{V=1}| / |evaluated|
    Label source: LLM verdict V stored in recommendations.verdict.

    Args:
        verdicts: Sequence of binary verdict values (0 or 1).

    Returns:
        TVR in [0.0, 1.0]. Returns 0.0 when verdicts is empty.
    """
    if not verdicts:
        return 0.0
    return sum(verdicts) / len(verdicts)


def sld(
    semantic_scores: Sequence[float],
    logical_scores: Sequence[float],
    tau_s: float,
    tau_l: float,
) -> float:
    """Score-Label Discordance: fraction of pairs where S and L thresholds disagree.

    Formula (methods.md):
        SLD = |{I[S >= tau_S] != I[L >= tau_L]}| / |evaluated|

    Label source: semantic_score and logical_score from recommendations table.
    No human GT required -- measures internal score disagreement only.

    Args:
        semantic_scores: Semantic scores per pair.
        logical_scores: Logical scores per pair (same length as semantic_scores).
        tau_s: Semantic threshold (COMPAT_TAU_S).
        tau_l: Logical threshold (COMPAT_TAU_L).

    Returns:
        SLD in [0.0, 1.0]. Returns 0.0 when inputs are empty.
    """
    n = len(semantic_scores)
    if n == 0:
        return 0.0
    discordant = sum(
        1
        for s, lg in zip(semantic_scores, logical_scores)
        if (s >= tau_s) != (lg >= tau_l)
    )
    return discordant / n


def hsv(
    hybrid_scores: Sequence[float] | np.ndarray,
    verdicts: Sequence[int] | np.ndarray,
    tau_candidates: Sequence[float] | None = None,
) -> float:
    """Hybrid-Score Validity: max agreement between sign(score >= tau) and verdict V.

    Formula (methods.md):
        HSV = max_{tau} agreement(I[hybrid_score >= tau], V)
    where agreement = fraction of pairs where threshold indicator matches verdict.

    Label source: LLM verdict V from recommendations.verdict (label-free vs human GT).

    Vectorized via numpy: broadcasts scores[:,None] >= tau[None,:] to produce an
    (N, T) boolean agreement matrix, then takes column-wise mean and picks the max.

    Args:
        hybrid_scores: Hybrid scores per pair.
        verdicts: Binary LLM verdicts per pair (0 or 1).
        tau_candidates: Optional explicit grid of threshold values to sweep.
            Defaults to 101 evenly-spaced values in [0, 1].

    Returns:
        HSV in [0.0, 1.0]. Returns 0.0 when inputs are empty.
    """
    n = len(hybrid_scores)
    if n == 0:
        return 0.0

    scores_arr = np.asarray(hybrid_scores, dtype=np.float64)
    verdicts_arr = np.asarray(verdicts, dtype=np.int8)

    if tau_candidates is None:
        tau_arr = _TAU_DEFAULT
    else:
        tau_arr = np.asarray(tau_candidates, dtype=np.float64)

    # Broadcast: (N, T) -- scores[:,None] >= tau[None,:]
    predicted = scores_arr[:, None] >= tau_arr[None, :]  # shape (N, T)
    labels = (verdicts_arr == 1)[:, None]  # shape (N, 1) -> broadcast to (N, T)
    agreement = np.mean(predicted == labels, axis=0)  # shape (T,)
    return float(agreement.max())


# ---------------------------------------------------------------------------
# Alpha sweep: returns per-alpha metrics curve + optimal alpha
# ---------------------------------------------------------------------------


def alpha_sweep(
    rows: list[dict],
    tau_s: float,
    tau_l: float,
    k: int = 10,
    gt_map: dict[int, set[int]] | None = None,
) -> tuple[list[dict], float]:
    """Sweep alpha in {0.00, 0.05, ..., 1.00}, compute per-alpha HSV (and GT metrics if available).

    For each alpha value, recomputes hybrid_score = alpha*S + (1-alpha)*L for every
    recommendation row, then computes HSV against the stored LLM verdict V.
    When ground-truth is provided (gt_map), also computes Precision/Recall/F1/NDCG@K.

    Label source for HSV: LLM verdict V (label-free).
    Label source for P/R/F1/NDCG: human ground-truth gt_map (when provided).

    Vectorized via numpy for the HSV + bootstrap hot path:
    - S, L, V held as numpy arrays (N,).
    - Per-alpha hybrid scores computed with a single multiply-add op.
    - HSV computed via vectorized broadcast (N, T) agreement matrix.
    - Bootstrap resampling done ONCE via _make_boot_indices() and reused across
      all 21 alpha values -- the resample indices are alpha-independent.
      This avoids regenerating B=1000 * N random integers 21 times.

    Args:
        rows: List of dicts with keys: product_id (int), recommended_id (int),
              semantic_score (float), logical_score (float), verdict (int).
        tau_s: Semantic threshold (used only for verdict recomputation; not used in
               alpha_sweep itself since we use stored V as the target label).
        tau_l: Logical threshold (same note).
        k: Cutoff for GT-dependent metrics (default 10).
        gt_map: Optional dict mapping source product_id -> set of relevant recommended_ids.

    Returns:
        Tuple of (curve, alpha_star) where:
        - curve is a list of dicts, one per alpha, with keys:
          alpha_value, hsv, precision_at_k, recall_at_k, f1_at_k, ndcg_at_k,
          sample_size, ci_lower, ci_upper.
        - alpha_star is the alpha with the highest HSV (first maximum if tie).
    """
    if not rows:
        return [], 0.6  # default alpha per settings

    # --- Precompute numpy arrays for the hot path ---
    s_arr = np.array([float(r["semantic_score"]) for r in rows], dtype=np.float64)
    l_arr = np.array([float(r["logical_score"]) for r in rows], dtype=np.float64)
    v_arr = np.array([int(r["verdict"]) for r in rows], dtype=np.int8)
    n = len(rows)

    # Pre-generate resample indices ONCE: shape (B, n), shared across all 21 alpha.
    # Pre-resample s and l base arrays (float32 to halve memory) and the verdict
    # label (uint8) -- these are all alpha-independent.
    # Memory: (1000, 33854) * (4 + 4 + 1) = ~289 MB total for the three boot matrices.
    boot_idx = _make_boot_indices(n, n_boot=1000)  # (B, n) int64
    s_boot = s_arr[boot_idx].astype(np.float32)  # (B, n) float32, reused each alpha
    l_boot = l_arr[boot_idx].astype(np.float32)  # (B, n) float32, reused each alpha
    lbl_boot = (v_arr == 1)[boot_idx].view(np.uint8)  # (B, n) uint8, fixed labels

    # Pre-allocate per-tau working buffers (reused in the inner loop)
    b = boot_idx.shape[0]
    _pred_buf = np.empty((b, n), dtype=np.uint8)
    _agree_buf = np.empty(b, dtype=np.float64)
    _inv_n = 1.0 / n

    # Group by source product for ranking metrics (GT path -- not the bottleneck)
    from collections import defaultdict

    grouped: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        grouped[int(r["product_id"])].append(r)

    curve: list[dict] = []
    best_hsv = -1.0
    alpha_star = 0.6

    for alpha in ALPHA_GRID:
        alpha_f = float(alpha)
        alpha_f32 = np.float32(alpha_f)

        # Recompute hybrid score for this alpha -- single numpy op over all N pairs
        hybrid_arr = alpha_f * s_arr + (1.0 - alpha_f) * l_arr

        # HSV via vectorized function
        _hsv = hsv(hybrid_arr, v_arr)

        # GT-dependent metrics (only when gt_map provided and non-empty) -- pure Python
        p_k = r_k = f_k = n_k = None
        if gt_map:
            reranked: dict[int, list[dict]] = {}
            for pid, group in grouped.items():
                ranked = sorted(
                    group,
                    key=lambda r: (
                        alpha_f * float(r["semantic_score"])
                        + (1.0 - alpha_f) * float(r["logical_score"])
                    ),
                    reverse=True,
                )
                reranked[pid] = ranked

            p_vals, r_vals, f_vals, n_vals = [], [], [], []
            for pid, ranked_group in reranked.items():
                relevant = gt_map.get(pid, set())
                if not relevant:
                    continue
                ranked_ids = [int(r["recommended_id"]) for r in ranked_group]
                p_vals.append(precision_at_k(relevant, ranked_ids, k))
                r_vals.append(recall_at_k(relevant, ranked_ids, k))
                f_vals.append(f1_at_k(relevant, ranked_ids, k))
                n_vals.append(ndcg_at_k(relevant, ranked_ids, k))
            if p_vals:
                p_k = sum(p_vals) / len(p_vals)
                r_k = sum(r_vals) / len(r_vals)
                f_k = sum(f_vals) / len(f_vals)
                n_k = sum(n_vals) / len(n_vals)

        # Bootstrap CI:
        # hybrid boot scores = alpha * s_boot + (1-alpha) * l_boot (in-place float32)
        h_boot = alpha_f32 * s_boot + np.float32(1.0 - alpha_f) * l_boot  # (B, n) f32
        ci_lower, ci_upper = _bootstrap_ci_hsv_precomputed(
            h_boot, lbl_boot, _pred_buf, _agree_buf, _inv_n
        )

        entry: dict = {
            "alpha_value": alpha,
            "hsv": _hsv,
            "precision_at_k": p_k,
            "recall_at_k": r_k,
            "f1_at_k": f_k,
            "ndcg_at_k": n_k,
            "sample_size": n,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        curve.append(entry)

        if _hsv > best_hsv:
            best_hsv = _hsv
            alpha_star = alpha

    return curve, alpha_star


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _bootstrap_ci_hsv_precomputed(
    h_boot: np.ndarray,
    lbl_boot: np.ndarray,
    pred_buf: np.ndarray,
    agree_buf: np.ndarray,
    inv_n: float,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Fast bootstrap CI for HSV given pre-resampled score and label matrices.

    Called from alpha_sweep() where h_boot and lbl_boot are already materialised
    for the current alpha. Pre-allocated buffers pred_buf and agree_buf avoid
    per-tau heap allocations in the inner loop.

    Args:
        h_boot: Pre-resampled hybrid scores, shape (B, N), dtype float32.
        lbl_boot: Pre-resampled verdict labels, shape (B, N), dtype uint8 (0/1).
        pred_buf: Pre-allocated work buffer, shape (B, N), dtype uint8.
        agree_buf: Pre-allocated work buffer, shape (B,), dtype float64.
        inv_n: 1.0 / N (pre-computed to avoid per-iteration division).
        confidence: Confidence level (default 0.95).

    Returns:
        (ci_lower, ci_upper) tuple.
    """
    b = h_boot.shape[0]
    boot_max = np.zeros(b, dtype=np.float64)

    for tau in _TAU_DEFAULT:
        tau_f32 = np.float32(tau)
        np.greater_equal(h_boot, tau_f32, out=pred_buf)
        np.equal(pred_buf, lbl_boot, out=pred_buf)
        np.sum(pred_buf, axis=1, dtype=np.float64, out=agree_buf)
        agree_buf *= inv_n
        np.maximum(boot_max, agree_buf, out=boot_max)

    boot_max.sort()
    alpha_tail = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha_tail * b)))
    hi_idx = min(b - 1, int(math.ceil((1.0 - alpha_tail) * b)) - 1)
    return (float(boot_max[lo_idx]), float(boot_max[hi_idx]))


def _make_boot_indices(n: int, n_boot: int = 1000) -> np.ndarray:
    """Generate bootstrap resample indices using the fixed BOOTSTRAP_SEED.

    Args:
        n: Sample size (number of pairs).
        n_boot: Number of bootstrap resamples.

    Returns:
        Integer index array of shape (n_boot, n), dtype int64.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(0, n, size=(n_boot, n))


def _bootstrap_ci_hsv(
    hybrid_scores: np.ndarray | Sequence[float],
    verdicts: np.ndarray | Sequence[int],
    n_boot: int = 1000,
    confidence: float = 0.95,
    boot_idx: np.ndarray | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI for HSV using fixed seed BOOTSTRAP_SEED=42.

    Vectorized via numpy. Algorithm:
    1. Resample scores as float32 (B, N) and labels as uint8 (B, N) -- ~170 MB total
       for B=1000, N=33854, vs ~542 MB with float64.
    2. Sweep 101 tau values in a Python loop; each step is a (B, N) numpy comparison
       with pre-allocated output buffers to minimise allocations.
    3. Track running per-resample HSV max with np.maximum(out=).

    The numpy RNG produces different draws than the old random.Random
    implementation, so the exact CI bounds differ from the pure-Python version,
    but the statistical validity is identical (same seed, same B=1000, same
    percentile method).

    Args:
        hybrid_scores: Hybrid scores (already computed for a fixed alpha).
            Accepts numpy array or any Sequence[float].
        verdicts: Binary LLM verdicts. Accepts numpy array or any Sequence[int].
        n_boot: Number of bootstrap resamples (B=1000 per methods.md section 5).
        confidence: Confidence level (default 0.95).
        boot_idx: Optional pre-generated resample indices of shape (n_boot, n).
            When provided, n_boot is ignored (indices determine B). Passing
            pre-generated indices from alpha_sweep() avoids redundant RNG calls.

    Returns:
        (ci_lower, ci_upper) tuple.
    """
    scores_arr = np.asarray(hybrid_scores, dtype=np.float64)
    verdicts_arr = np.asarray(verdicts, dtype=np.int8)
    n = len(scores_arr)
    if n == 0:
        return (0.0, 0.0)

    # Use provided indices or generate fresh ones
    if boot_idx is None:
        idx = _make_boot_indices(n, n_boot)
    else:
        idx = boot_idx

    b = idx.shape[0]

    # Resample: use float32 for scores (halves memory vs float64), uint8 for labels.
    # Peak memory: b*n*(4+1) bytes = 1000*33854*5 = ~161 MB for the two boot matrices.
    s_boot = scores_arr[idx].astype(np.float32)  # (B, N) float32
    l_boot = (verdicts_arr == 1)[idx].view(np.uint8)  # (B, N) uint8

    # Pre-allocate working buffers to avoid per-tau allocation in the loop.
    predicted = np.empty((b, n), dtype=np.uint8)  # reused each tau iteration
    agree_buf = np.empty(b, dtype=np.float64)
    boot_max = np.zeros(b, dtype=np.float64)

    inv_n = 1.0 / n

    for tau in _TAU_DEFAULT:
        tau_f32 = np.float32(tau)
        # In-place: predicted[i,j] = 1 if s_boot[i,j] >= tau else 0
        np.greater_equal(s_boot, tau_f32, out=predicted)
        # agreement per resample: mean over N of (predicted == l_boot)
        # np.equal writes into predicted (reuse buffer); then sum axis=1
        np.equal(predicted, l_boot, out=predicted)
        np.sum(predicted, axis=1, dtype=np.float64, out=agree_buf)
        agree_buf *= inv_n
        np.maximum(boot_max, agree_buf, out=boot_max)

    boot_max.sort()
    alpha_tail = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha_tail * b)))
    hi_idx = min(b - 1, int(math.ceil((1.0 - alpha_tail) * b)) - 1)
    return (float(boot_max[lo_idx]), float(boot_max[hi_idx]))


# ---------------------------------------------------------------------------
# Cohen's kappa (requires human GT + LLM verdict)
# ---------------------------------------------------------------------------


def cohen_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float | None:
    """Cohen's kappa between human ground-truth labels and LLM verdicts.

    Formula (methods.md): kappa = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.

    Args:
        y_true: Human ground-truth binary labels (0 or 1).
        y_pred: LLM verdict binary predictions (0 or 1).

    Returns:
        Cohen's kappa in [-1.0, 1.0]. Returns None when inputs are empty or
        when expected agreement is 1.0 (all same class -- undefined kappa).
    """
    n = len(y_true)
    if n == 0 or len(y_pred) != n:
        return None

    # 2x2 confusion matrix counts
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)

    p_o = (tp + tn) / n

    # Marginal probabilities
    p_pos_true = (tp + fn) / n
    p_neg_true = (tn + fp) / n
    p_pos_pred = (tp + fp) / n
    p_neg_pred = (tn + fn) / n

    p_e = p_pos_true * p_pos_pred + p_neg_true * p_neg_pred
    if abs(1.0 - p_e) < 1e-12:
        return None  # undefined: all same class in one set

    return (p_o - p_e) / (1.0 - p_e)


# ---------------------------------------------------------------------------
# Bootstrap CI (public, for external use)
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 500,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap percentile 95% CI for the mean of a scalar metric.

    Uses BOOTSTRAP_SEED=42 for reproducibility. Uses numpy RNG internally
    (see module docstring for the RNG-change note vs the old random.Random version).

    Args:
        values: Observed scalar values (e.g., per-query NDCG values).
        n_boot: Number of bootstrap resamples.
        confidence: Confidence level (default 0.95).

    Returns:
        (ci_lower, ci_upper) tuple. Returns (0.0, 0.0) when values is empty.
    """
    n = len(values)
    if n == 0:
        return (0.0, 0.0)

    vals_arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Generate all resample index matrices at once: shape (n_boot, n)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = vals_arr[idx]  # shape (n_boot, n)
    boot_means = samples.mean(axis=1)  # shape (n_boot,)
    boot_means.sort()

    alpha_tail = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha_tail * n_boot)))
    hi_idx = min(n_boot - 1, int(math.ceil((1.0 - alpha_tail) * n_boot)) - 1)
    return (float(boot_means[lo_idx]), float(boot_means[hi_idx]))
