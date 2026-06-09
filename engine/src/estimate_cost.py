"""Zero-cost preflight estimator for the AVTC baseline compatibility experiment.

Mirrors the EXACT pipeline order from pipeline.py:
  fetch source from Typesense -> retrieve_candidates -> filter_candidates
  -> keep _score >= COMPAT_TAU_S -> batch by COMPAT_BATCH_SIZE -> build prompt
  -> count tokens with tiktoken (STOP HERE - no LLM/OpenAI calls ever).

Run inside the engine container:
    python -m estimate_cost
    python -m estimate_cost --sample 200
    python -m estimate_cost --sample 500 --price-in 0.075 --price-out 0.30

All token counts are real (tiktoken on actual prompts built from live Typesense data).
No LLM, no OpenAI, no mock invocations - zero spend guaranteed.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import statistics
from datetime import date
from pathlib import Path

import aiomysql
import httpx
import tiktoken

from app.core.config import settings
from app.services.llm.verifier import SYSTEM_PROMPT, build_user_prompt
from app.services.retrieval.candidates import filter_candidates, retrieve_candidates

log = logging.getLogger(__name__)

# Output token model: what the LLM returns per verified candidate.
# Schema per verifier.RESPONSE_FORMAT (8 integer/bool/float/string fields):
#   candidate_id, verdict, logical_score, context_code, rules_evaluated,
#   rules_passed, rules_failed, rules_undefined
# "lean" = short context_code (~2 words), no extra whitespace in JSON.
# "verbose" = longer context_code string + JSON whitespace overhead.
OUTPUT_TOKENS_PER_CANDIDATE_LEAN = 28
OUTPUT_TOKENS_PER_CANDIDATE_VERBOSE = 55

# Fixed overhead per batch response: JSON envelope {"results":[...]}
OUTPUT_TOKENS_ENVELOPE = 10

# Schema overhead added to every prompt for the response_format JSON schema
# (the schema definition is sent as part of the API request metadata, not the
# prompt itself - but we add a small conservative buffer for any system overhead).
SCHEMA_OVERHEAD_TOKENS = 40

# Container path where /docs is mounted (./project/docs on the host)
DOCS_MOUNT = Path("/docs")
REPORT_PATH = DOCS_MOUNT / "article2" / "cost-preflight.md"


async def _fetch_active_product_ids(sample: int) -> list[int]:
    """Return a deterministically spaced sample of active product_ids from catalog DB.

    Uses ORDER BY product_id ASC and picks every K-th row so the sample is
    spread evenly across the full ID range. No randomness - same sample every run.

    Args:
        sample: Target number of product IDs to return.

    Returns:
        Sorted list of product_ids.
    """
    conn = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    try:
        async with conn.cursor() as cur:
            # Total active count
            await cur.execute("SELECT COUNT(*) FROM products WHERE status = %s", (1,))
            (total_active,) = await cur.fetchone()

            # All active product_ids ordered ascending
            await cur.execute(
                "SELECT product_id FROM products WHERE status = %s ORDER BY product_id ASC",
                (1,),
            )
            rows = await cur.fetchall()
    finally:
        conn.close()

    all_ids = [r[0] for r in rows]
    n = len(all_ids)

    if sample >= n:
        log.info("sample=%d >= total_active=%d - using all", sample, n)
        return all_ids, n

    # Deterministic even spread: pick every K-th id
    step = n / sample
    sampled = [all_ids[int(i * step)] for i in range(sample)]
    log.info("sampled %d products from %d total (step=%.2f)", len(sampled), n, step)
    return sampled, n


async def _fetch_source_doc(
    http: httpx.AsyncClient,
    product_id: int,
) -> dict | None:
    """Fetch one product document (with embedding) from Typesense.

    Mirrors pipeline._fetch_source exactly.

    Args:
        http: Shared async HTTP client.
        product_id: Product to fetch.

    Returns:
        Document dict or None if not found.
    """
    ts_base = settings.typesense_base_url
    ts_key = settings.TYPESENSE_API_KEY
    ts_col = settings.TYPESENSE_COLLECTION

    url = f"{ts_base}/collections/{ts_col}/documents/search"
    params = {
        "q": "*",
        "query_by": "name",
        "filter_by": f"product_id:={product_id}",
        "per_page": 1,
    }
    resp = await http.get(url, params=params, headers={"X-TYPESENSE-API-KEY": ts_key})
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    return hits[0]["document"] if hits else None


def _count_input_tokens(
    enc: tiktoken.Encoding,
    source: dict,
    batch: list[dict],
) -> int:
    """Count input tokens for one batch: SYSTEM_PROMPT + user prompt + schema overhead.

    Args:
        enc: tiktoken encoding instance.
        source: Source product document.
        batch: Candidate batch (up to COMPAT_BATCH_SIZE items).

    Returns:
        Total input token count for this batch.
    """
    user_prompt = build_user_prompt(source, batch)
    system_tokens = len(enc.encode(SYSTEM_PROMPT))
    user_tokens = len(enc.encode(user_prompt))
    return system_tokens + user_tokens + SCHEMA_OVERHEAD_TOKENS


def _estimate_output_tokens(batch_size: int) -> tuple[int, int]:
    """Estimate output token count for one batch (lean and verbose).

    Args:
        batch_size: Number of candidates in this batch.

    Returns:
        Tuple of (lean_tokens, verbose_tokens) both capped at MAX_TOKENS.
    """
    lean = OUTPUT_TOKENS_ENVELOPE + batch_size * OUTPUT_TOKENS_PER_CANDIDATE_LEAN
    verbose = OUTPUT_TOKENS_ENVELOPE + batch_size * OUTPUT_TOKENS_PER_CANDIDATE_VERBOSE
    cap = settings.MAX_TOKENS
    return min(lean, cap), min(verbose, cap)


async def run_estimation(sample: int) -> dict:
    """Run the full zero-cost estimation pipeline.

    Mirrors pipeline.py exactly up to (but not including) llm.verify.
    No LLM or OpenAI SDK objects are created or called.

    Args:
        sample: Number of products to sample.

    Returns:
        Dict with all projection statistics.
    """
    # Encoding setup - log which one is used
    try:
        enc = tiktoken.encoding_for_model(settings.PRIMARY_MODEL)
        enc_name = f"{settings.PRIMARY_MODEL} -> {enc.name}"
    except KeyError:
        enc = tiktoken.get_encoding("o200k_base")
        enc_name = f"fallback o200k_base (model '{settings.PRIMARY_MODEL}' not in tiktoken registry)"
    log.info("tiktoken encoding: %s", enc_name)

    product_ids, total_active = await _fetch_active_product_ids(sample)
    actual_sample = len(product_ids)
    scale_factor = total_active / actual_sample

    ts_base = settings.typesense_base_url
    ts_key = settings.TYPESENSE_API_KEY
    ts_col = settings.TYPESENSE_COLLECTION

    # Per-product accumulators
    candidates_kept: list[int] = []
    batches_per_product: list[int] = []
    total_input_tokens = 0
    total_output_tokens_lean = 0
    total_output_tokens_verbose = 0
    total_batch_requests = 0
    skipped_no_source = 0
    skipped_no_candidates = 0

    http = httpx.AsyncClient(timeout=60.0)
    try:
        for idx, pid in enumerate(product_ids, 1):
            if idx % 50 == 0 or idx == actual_sample:
                log.info("progress: %d/%d products processed", idx, actual_sample)

            # Step 1: fetch source doc from Typesense (mirrors pipeline._fetch_source)
            source = await _fetch_source_doc(http, pid)
            if not source:
                skipped_no_source += 1
                log.debug("product=%d: no source doc in Typesense", pid)
                continue

            # Step 2: retrieve_candidates (semantic strategy, top_k=COMPAT_TOP_K)
            raw = await retrieve_candidates(
                settings.RETRIEVAL_STRATEGY,
                http,
                ts_base,
                ts_key,
                ts_col,
                source,
                settings.COMPAT_TOP_K,
            )

            # Step 3: filter_candidates (drop same-category + self)
            filtered = filter_candidates(source, raw)

            # Step 4: keep _score >= COMPAT_TAU_S
            cands = [
                c
                for c in filtered
                if float(c.get("_score", 0.0)) >= settings.COMPAT_TAU_S
            ]

            if not cands:
                skipped_no_candidates += 1
                log.debug("product=%d: no candidates after tau_s filter", pid)
                continue

            candidates_kept.append(len(cands))

            # Step 5: batch by COMPAT_BATCH_SIZE and count tokens (STOP before llm.verify)
            bs = settings.COMPAT_BATCH_SIZE
            n_batches = math.ceil(len(cands) / bs)
            batches_per_product.append(n_batches)

            for k in range(0, len(cands), bs):
                chunk = cands[k : k + bs]
                in_tok = _count_input_tokens(enc, source, chunk)
                out_lean, out_verbose = _estimate_output_tokens(len(chunk))

                total_input_tokens += in_tok
                total_output_tokens_lean += out_lean
                total_output_tokens_verbose += out_verbose
                total_batch_requests += 1

    finally:
        await http.aclose()

    # Sample-level stats
    avg_cands = statistics.mean(candidates_kept) if candidates_kept else 0.0
    med_cands = statistics.median(candidates_kept) if candidates_kept else 0.0
    min_cands = min(candidates_kept) if candidates_kept else 0
    max_cands = max(candidates_kept) if candidates_kept else 0
    avg_batches = statistics.mean(batches_per_product) if batches_per_product else 0.0

    products_with_candidates = len(candidates_kept)

    # Extrapolate to full run
    proj_batch_requests = round(total_batch_requests * scale_factor)
    proj_input_tokens = round(total_input_tokens * scale_factor)
    proj_output_lean = round(total_output_tokens_lean * scale_factor)
    proj_output_verbose = round(total_output_tokens_verbose * scale_factor)

    return {
        # metadata
        "encoding": enc_name,
        "primary_model": settings.PRIMARY_MODEL,
        "retrieval_strategy": settings.RETRIEVAL_STRATEGY,
        "compat_top_k": settings.COMPAT_TOP_K,
        "compat_tau_s": settings.COMPAT_TAU_S,
        "compat_batch_size": settings.COMPAT_BATCH_SIZE,
        "max_tokens": settings.MAX_TOKENS,
        # sample info
        "sample_size": actual_sample,
        "total_active": total_active,
        "scale_factor": scale_factor,
        "skipped_no_source": skipped_no_source,
        "skipped_no_candidates": skipped_no_candidates,
        "products_with_candidates": products_with_candidates,
        # candidate distribution
        "avg_candidates": avg_cands,
        "median_candidates": med_cands,
        "min_candidates": min_cands,
        "max_candidates": max_cands,
        "avg_batches_per_product": avg_batches,
        # sample raw totals
        "sample_batch_requests": total_batch_requests,
        "sample_input_tokens": total_input_tokens,
        "sample_output_lean": total_output_tokens_lean,
        "sample_output_verbose": total_output_tokens_verbose,
        # projected full-run totals
        "proj_batch_requests": proj_batch_requests,
        "proj_input_tokens": proj_input_tokens,
        "proj_output_lean": proj_output_lean,
        "proj_output_verbose": proj_output_verbose,
        # output token assumptions
        "output_tokens_per_candidate_lean": OUTPUT_TOKENS_PER_CANDIDATE_LEAN,
        "output_tokens_per_candidate_verbose": OUTPUT_TOKENS_PER_CANDIDATE_VERBOSE,
    }


def _cost_table(
    proj_input: int,
    proj_out_lean: int,
    proj_out_verbose: int,
    price_in: float,
    price_out: float,
) -> str:
    """Build a 4-row cost table (standard/batch x lean/verbose).

    Args:
        proj_input: Projected total input tokens.
        proj_out_lean: Projected total output tokens (lean).
        proj_out_verbose: Projected total output tokens (verbose).
        price_in: USD per 1M input tokens.
        price_out: USD per 1M output tokens.

    Returns:
        Formatted multi-line string.
    """

    def cost(in_tok: int, out_tok: int, in_p: float, out_p: float) -> float:
        return (in_tok / 1_000_000) * in_p + (out_tok / 1_000_000) * out_p

    rows = []
    for label, in_p, out_p in [
        ("Standard (lean output)", price_in, price_out),
        ("Standard (verbose output)", price_in, price_out),
        ("Batch API -50% (lean output)", price_in * 0.5, price_out * 0.5),
        ("Batch API -50% (verbose output)", price_in * 0.5, price_out * 0.5),
    ]:
        if "lean" in label:
            out_tok = proj_out_lean
        else:
            out_tok = proj_out_verbose
        c = cost(proj_input, out_tok, in_p, out_p)
        rows.append(f"  {label:<42} ${c:>10.2f}")

    return "\n".join(rows)


def _print_summary(r: dict, price_in: float, price_out: float) -> None:
    """Print the clean summary block to stdout.

    Args:
        r: Result dict from run_estimation.
        price_in: USD per 1M input tokens (0.0 = placeholder).
        price_out: USD per 1M output tokens (0.0 = placeholder).
    """
    prices_are_placeholder = price_in == 0.0 and price_out == 0.0
    price_note = (
        "\n  *** PRICES ARE PLACEHOLDER (0.0). Supply real gpt-5-nano prices via\n"
        "  *** --price-in and --price-out (USD per 1M tokens) to get real cost estimates."
        if prices_are_placeholder
        else f"\n  Prices supplied: in=${price_in}/1M  out=${price_out}/1M"
    )

    print(
        f"""
=============================================================
  AVTC Baseline Experiment - Cost Pre-Flight Estimation
  Date: {date.today()}
=============================================================

--- Configuration ---
  Primary model     : {r["primary_model"]}
  Tiktoken encoding : {r["encoding"]}
  Retrieval strategy: {r["retrieval_strategy"]}
  COMPAT_TOP_K      : {r["compat_top_k"]}
  COMPAT_TAU_S      : {r["compat_tau_s"]}
  COMPAT_BATCH_SIZE : {r["compat_batch_size"]}
  MAX_TOKENS (output cap): {r["max_tokens"]}

--- Sample ---
  Sample size     : {r["sample_size"]:,}
  Total active    : {r["total_active"]:,}
  Scale factor    : {r["scale_factor"]:.4f}x
  Skipped (no Typesense doc) : {r["skipped_no_source"]}
  Skipped (no candidates after tau_s): {r["skipped_no_candidates"]}
  Products with candidates in sample : {r["products_with_candidates"]}

--- Candidate Distribution (after filter + tau_s) ---
  Average  : {r["avg_candidates"]:.2f}
  Median   : {r["median_candidates"]:.1f}
  Min      : {r["min_candidates"]}
  Max      : {r["max_candidates"]}
  Avg batches per product: {r["avg_batches_per_product"]:.2f}

--- Output Token Assumptions ---
  Per candidate lean    : {r["output_tokens_per_candidate_lean"]} tokens
    (candidate_id + verdict + logical_score + context_code short + 4 rule ints)
  Per candidate verbose : {r["output_tokens_per_candidate_verbose"]} tokens
    (same fields + longer context_code + JSON whitespace)
  Schema/envelope overhead per batch: {OUTPUT_TOKENS_ENVELOPE} tokens
  SCHEMA_OVERHEAD_TOKENS added to input per batch: {SCHEMA_OVERHEAD_TOKENS}

--- Sample Raw Totals ---
  Batch requests  : {r["sample_batch_requests"]:,}
  Input tokens    : {r["sample_input_tokens"]:,}
  Output (lean)   : {r["sample_output_lean"]:,}
  Output (verbose): {r["sample_output_verbose"]:,}

--- Projected Full Run ({r["total_active"]:,} products) ---
  Batch API requests : {r["proj_batch_requests"]:,}
  Input tokens       : {r["proj_input_tokens"]:,}
  Output tokens lean : {r["proj_output_lean"]:,}
  Output tokens verbose: {r["proj_output_verbose"]:,}

--- Cost Projection (Batch API = -50% off standard) ---
{price_note}

{_cost_table(r["proj_input_tokens"], r["proj_output_lean"], r["proj_output_verbose"], price_in, price_out)}

=============================================================
  ZERO LLM/OpenAI calls made. Pure Typesense + tiktoken.
  No RealLLM, no MockLLM, no OpenAI SDK invoked.
=============================================================
"""
    )


def _write_report(r: dict, price_in: float, price_out: float) -> None:
    """Write the cost preflight report to /docs/article2/cost-preflight.md.

    Args:
        r: Result dict from run_estimation.
        price_in: USD per 1M input tokens.
        price_out: USD per 1M output tokens.
    """
    prices_are_placeholder = price_in == 0.0 and price_out == 0.0
    price_note = (
        "> **PRICES ARE PLACEHOLDER (0.0)**. Supply real gpt-5-nano prices to get"
        " real cost estimates."
        if prices_are_placeholder
        else f"Prices used: in=${price_in}/1M out=${price_out}/1M"
    )

    def cost(in_tok: int, out_tok: int, in_p: float, out_p: float) -> float:
        return (in_tok / 1_000_000) * in_p + (out_tok / 1_000_000) * out_p

    def fmt_cost(c: float) -> str:
        return "PLACEHOLDER" if prices_are_placeholder else f"${c:.2f}"

    rows_md = []
    for label, in_p, out_p, out_tok in [
        (
            "Standard - lean",
            price_in,
            price_out,
            r["proj_output_lean"],
        ),
        (
            "Standard - verbose",
            price_in,
            price_out,
            r["proj_output_verbose"],
        ),
        (
            "Batch API -50% - lean",
            price_in * 0.5,
            price_out * 0.5,
            r["proj_output_lean"],
        ),
        (
            "Batch API -50% - verbose",
            price_in * 0.5,
            price_out * 0.5,
            r["proj_output_verbose"],
        ),
    ]:
        c = cost(r["proj_input_tokens"], out_tok, in_p, out_p)
        rows_md.append(f"| {label} | {fmt_cost(c)} |")

    cost_table_md = "\n".join(rows_md)

    content = f"""# AVTC Baseline Experiment - Cost Pre-Flight Estimation

**Date:** {date.today()}
**Author:** estimate_cost.py (zero-cost preflight)
**Status:** Draft
**Version:** 1.0

---

## Summary

Zero-cost pre-flight token and cost projection for the AVTC baseline compatibility
experiment (all 14,767 active products, model: `{r["primary_model"]}`, Batch API).

No LLM calls were made. This report is based on real Typesense vector retrieval +
tiktoken token counting on actual prompts built from live catalog data.

---

## Configuration

| Parameter | Value |
|---|---|
| Primary model | `{r["primary_model"]}` |
| Tiktoken encoding | `{r["encoding"]}` |
| Retrieval strategy | `{r["retrieval_strategy"]}` |
| COMPAT_TOP_K | {r["compat_top_k"]} |
| COMPAT_TAU_S | {r["compat_tau_s"]} |
| COMPAT_BATCH_SIZE | {r["compat_batch_size"]} |
| MAX_TOKENS (output cap) | {r["max_tokens"]} |

---

## Sample

| Metric | Value |
|---|---|
| Sample size | {r["sample_size"]:,} |
| Total active products | {r["total_active"]:,} |
| Scale factor | {r["scale_factor"]:.4f}x |
| Skipped (no Typesense doc) | {r["skipped_no_source"]} |
| Skipped (no candidates after tau_s) | {r["skipped_no_candidates"]} |
| Products with candidates | {r["products_with_candidates"]} |

Sampling method: deterministic even spread (every K-th product_id ordered
ascending). Same sample every run - no randomness.

---

## Candidate Distribution (after filter + tau_s >= {r["compat_tau_s"]})

| Metric | Value |
|---|---|
| Average | {r["avg_candidates"]:.2f} |
| Median | {r["median_candidates"]:.1f} |
| Min | {r["min_candidates"]} |
| Max | {r["max_candidates"]} |
| Avg batches per product | {r["avg_batches_per_product"]:.2f} |

---

## Output Token Assumptions

Each LLM response object contains 8 fields per candidate (from `RESPONSE_FORMAT`):
`candidate_id` (int), `verdict` (bool), `logical_score` (float),
`context_code` (str), `rules_evaluated` (int), `rules_passed` (int),
`rules_failed` (int), `rules_undefined` (int).

| Scenario | Tokens per candidate | Rationale |
|---|---|---|
| Lean | {r["output_tokens_per_candidate_lean"]} | Short context_code (~1 word), compact JSON |
| Verbose | {r["output_tokens_per_candidate_verbose"]} | Longer context_code + JSON whitespace |

JSON envelope overhead per batch: {OUTPUT_TOKENS_ENVELOPE} tokens.
Input schema overhead per batch: {SCHEMA_OVERHEAD_TOKENS} tokens.

All output estimates are capped at MAX_TOKENS={r["max_tokens"]}.

---

## Projected Full Run ({r["total_active"]:,} products)

| Metric | Value |
|---|---|
| Total Batch API requests | {r["proj_batch_requests"]:,} |
| Total input tokens | {r["proj_input_tokens"]:,} |
| Total output tokens (lean) | {r["proj_output_lean"]:,} |
| Total output tokens (verbose) | {r["proj_output_verbose"]:,} |

---

## Cost Projection

{price_note}

| Scenario | Estimated Cost |
|---|---|
{cost_table_md}

Budget ceiling: ~$32 (Batch API).

---

## Assumptions and Caveats

1. Scale assumes the 500-product sample is representative of the full catalog.
   Deterministic spread sampling minimizes selection bias.
2. Products with no Typesense document or no passing candidates are excluded
   from both the token estimate and the cost projection.
3. Output token counts are estimates based on the JSON schema structure.
   Real output may vary by model verbosity.
4. Input token counts are exact (tiktoken on real prompts from live data).
5. The Batch API discount is -50% off standard pricing (OpenAI policy as of 2026-06).
6. gpt-5-nano pricing must be confirmed from the OpenAI pricing page before committing
   to the run. This document uses placeholder prices until confirmed.

---

## Verification

- ZERO OpenAI/LLM API calls made during estimation.
- No `RealLLM`, `MockLLM`, or `openai` SDK objects instantiated.
- Retrieval: real Typesense vector search on stored 1024-dim embeddings.
- Token counting: `tiktoken` (encoding: `{r["encoding"]}`).
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")
    log.info("report written to %s", REPORT_PATH)
    print(f"\nReport written to: {REPORT_PATH}")


async def _main(sample: int, price_in: float, price_out: float) -> None:
    """Async entry point.

    Args:
        sample: Number of products to sample.
        price_in: USD per 1M input tokens.
        price_out: USD per 1M output tokens.
    """
    log.info(
        "starting cost estimation: sample=%d price_in=%.4f price_out=%.4f",
        sample,
        price_in,
        price_out,
    )
    result = await run_estimation(sample)
    _print_summary(result, price_in, price_out)
    _write_report(result, price_in, price_out)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Zero-cost preflight token/cost estimator for AVTC baseline experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m estimate_cost\n"
            "  python -m estimate_cost --sample 200\n"
            "  python -m estimate_cost --sample 500 --price-in 0.075 --price-out 0.30\n"
            "\n"
            "No LLM calls are made. Pure Typesense + tiktoken."
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        help="Number of products to sample (default: 500). "
        "Deterministic spread across full catalog.",
    )
    parser.add_argument(
        "--price-in",
        type=float,
        default=0.0,
        metavar="USD_PER_1M",
        help="Input token price in USD per 1M tokens (default: 0.0 = placeholder). "
        "Supply real gpt-5-nano prices from the OpenAI pricing page.",
    )
    parser.add_argument(
        "--price-out",
        type=float,
        default=0.0,
        metavar="USD_PER_1M",
        help="Output token price in USD per 1M tokens (default: 0.0 = placeholder).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    asyncio.run(_main(args.sample, args.price_in, args.price_out))


if __name__ == "__main__":
    main()
