"""Per-product compatibility compute (engine side). The worker delegates here.

process_product: fetch the source (with embedding) from Typesense -> retrieve candidates
(strategy) -> filter same-category + self -> tau_S pre-filter -> batch by COMPAT_BATCH_SIZE
-> verify via the configured mode (jit: rule-based L; oneshot: legacy LLM verify)
-> hybrid score + verdict -> write recommendations + compatibility_evaluations.

COMPAT_MODE switch:
  jit      (default): get_rules per DISTINCT type-pair -> rule_eval per candidate
              -> aggregate_logical -> L. No LLM verify() calls.
  oneshot  (control arm): legacy llm.verify() path, unchanged for A/B comparison.

Pipeline instrumentation: RunContext opens a pipeline_runs row at entry and
flushes stage_metrics (retrieval, verification) at finish().
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from openai import AsyncOpenAI

from app.db.metrics_repository import MetricsRepository
from app.services.compatibility.cache import (
    MockRuleCache,
    RedisRuleCache,
    TableRuleCache,
)
from app.services.compatibility.ontology import get_rules
from app.services.compatibility.rule_eval import (
    aggregate_logical,
    compute_context_code,
    evaluate_rule,
)
from app.services.llm.verifier import get_llm
from app.services.metrics.run_context import RunContext
from app.services.retrieval.candidates import (
    _normalize_doc,
    filter_candidates,
    retrieve_candidates,
)
from app.services.scoring import compute_hybrid_score, compute_verdict

log = logging.getLogger(__name__)

# Pipeline version - increment when the compute graph changes.
_PIPELINE_VERSION = "1.1.0"


def _decide_verdict(strategy: str, semantic: float, logical: float, settings: Any) -> bool:
    """Strategy-aware compatibility verdict.

    cat_priors: the curated complement map already established that the candidate is a
    plausible accessory category, so the verdict rests on logical compatibility
    (L >= tau_L) alone. Semantic similarity to the source is intentionally LOW for
    complementary items (a phone is not similar to its case), so gating on S would
    discard exactly the accessories we want.

    Other strategies (semantic): classic V = I[S >= tau_S] * I[L >= tau_L].
    """
    if strategy == "cat_priors":
        return logical >= settings.COMPAT_TAU_L
    return compute_verdict(
        semantic, logical, settings.COMPAT_TAU_S, settings.COMPAT_TAU_L
    )


async def _fetch_source(
    http: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    collection: str,
    product_id: int,
) -> dict | None:
    """Fetch one product doc (with embedding) from Typesense."""
    url = f"{base_url}/collections/{collection}/documents/search"
    params = {
        "q": "*",
        "query_by": "name",
        "filter_by": f"product_id:={product_id}",
        "per_page": 1,
    }
    resp = await http.get(url, params=params, headers={"X-TYPESENSE-API-KEY": api_key})
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    return _normalize_doc(hits[0]["document"]) if hits else None


def _build_rule_cache(settings: Any, metrics_repo: MetricsRepository):
    """Build the appropriate RuleCache backend from settings.

    Args:
        settings: Engine Settings instance.
        metrics_repo: Connected MetricsRepository for TableRuleCache.

    Returns:
        A RuleCache implementation (MockRuleCache, TableRuleCache, or RedisRuleCache).
    """
    backend = getattr(settings, "RULE_CACHE_BACKEND", "mock")
    if settings.LLM_MODE == "mock" or backend == "mock":
        return MockRuleCache()
    if backend == "table":
        return TableRuleCache(metrics_repo)
    # redis (default for non-mock)
    try:
        import redis.asyncio as redis_async

        redis_client = redis_async.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False,
        )
        table_cache = TableRuleCache(metrics_repo)
        return RedisRuleCache(redis_client, table_cache=table_cache)
    except Exception as exc:
        log.warning("redis not available (%s), falling back to TableRuleCache", exc)
        return TableRuleCache(metrics_repo)


async def _verify_jit(
    source: dict,
    cands: list[dict],
    llm: Any,
    rule_cache: Any,
    settings: Any,
    stage_timer,
    attr_vocab: dict[str, list[str]] | None = None,
) -> list[dict]:
    """JIT-ontology verification path.

    For each DISTINCT (source_type, cand_type) pair, calls get_rules() once
    (cache hit: $0; cache miss: one llm.generate() call). Then evaluates
    rule_eval per candidate to compute L, context_code, and rule counts.

    When attr_vocab is supplied (built from product_ai_data at startup), the
    vocab is passed to get_rules() so rule-gen prompts reference REAL attribute
    keys. Without grounding, the LLM invents keys that never appear in
    normalized_json, producing rules_undefined=N/N and L=0 for every candidate.

    Args:
        source: Source product Typesense document.
        cands: Filtered candidate documents.
        llm: LLM client (real or mock) with generate() interface.
        rule_cache: RuleCache implementation.
        settings: Engine settings.
        stage_timer: StageTimer for the verification stage.
        attr_vocab: Optional per-type attribute key vocabulary from
            CatalogRepository.attribute_vocab_by_type(). None disables
            vocab-grounding (HDR gate still applies on cache hits if hdr_enabled).

    Returns:
        List of result dicts compatible with the scoring + write pipeline.
        Each dict has keys: candidate_id, logical_score, context_code,
        rules_evaluated, rules_passed, rules_failed, rules_undefined,
        evidence_claims, hallucinated_claims.
    """
    source_type = str(source.get("product_type") or "unknown")
    source_attrs = source.get("attributes") or {}

    # Collect DISTINCT type-pairs so rule-gen calls are minimised.
    distinct_pairs: set[tuple[str, str]] = set()
    for c in cands:
        ctype = str(c.get("product_type") or "unknown")
        distinct_pairs.add((source_type, ctype))

    # Fetch/generate rules for all distinct pairs (cache-aware).
    rules_by_pair: dict[tuple[str, str], tuple[list[dict], int, int]] = {}
    for type_a, type_b in distinct_pairs:
        # Resolve per-type vocab sets for HDR gate and prompt grounding.
        vocab_a: set[str] | None = None
        vocab_b: set[str] | None = None
        if attr_vocab:
            keys_a = attr_vocab.get(type_a)
            keys_b = attr_vocab.get(type_b)
            if keys_a:
                vocab_a = set(keys_a)
            if keys_b:
                vocab_b = set(keys_b)
        try:
            rules, ev, hall, was_miss = await get_rules(
                type_a,
                type_b,
                llm,
                rule_cache,
                hdr_enabled=getattr(settings, "HDR_ENABLED", True),
                vocab_a=vocab_a,
                vocab_b=vocab_b,
            )
            rules_by_pair[(type_a, type_b)] = (rules, ev, hall)
            # Increment llm_calls only on a real cache miss (actual LLM invocation).
            # Cache hits are $0 and must not inflate the JITOE metric.
            if was_miss:
                stage_timer.llm_calls += 1
        except Exception as exc:
            log.warning("rule-gen failed for (%s, %s): %s", type_a, type_b, exc)
            rules_by_pair[(type_a, type_b)] = ([], 0, 0)

    results: list[dict] = []
    l_agg = getattr(settings, "L_AGG", "weighted_product")

    for cand in cands:
        ctype = str(cand.get("product_type") or "unknown")
        pair_key = (source_type, ctype)
        rules, ev_claims, hall_claims = rules_by_pair.get(pair_key, ([], 0, 0))
        cand_attrs = cand.get("attributes") or {}

        eval_results: list[tuple[float, float, str]] = []
        for rule in rules:
            try:
                l_k, status = evaluate_rule(rule, source_attrs, cand_attrs)
            except Exception as exc:
                log.debug("rule eval error rule_id=%s: %s", rule.get("id"), exc)
                l_k, status = 0.0, "undefined"
            eval_results.append((l_k, float(rule.get("weight", 0.5)), status))

        big_l, n_passed, n_failed, n_undefined = aggregate_logical(eval_results, l_agg)
        ctx_code = compute_context_code(rules, source_type, ctype)

        results.append(
            {
                "candidate_id": cand.get("product_id"),
                "logical_score": big_l,
                "context_code": ctx_code,
                "rules_evaluated": len(rules),
                "rules_passed": n_passed,
                "rules_failed": n_failed,
                "rules_undefined": n_undefined,
                "evidence_claims": ev_claims,
                "hallucinated_claims": hall_claims,
            }
        )

    stage_timer.items_processed = len(cands)
    return results


async def process_product(
    product_id: int,
    experiment_id: str | None,
    settings: Any,
    metrics_repo: MetricsRepository,
    http: httpx.AsyncClient,
    attr_vocab: dict[str, list[str]] | None = None,
    strategy: str | None = None,
) -> dict:
    """Run the full per-product pipeline. Returns a summary dict.

    Raises LLMError (retryable) so the worker can retry/monitor until the LLM responds.
    Propagates httpx.HTTPError so the route handler can map it to a clean 503.

    Args:
        product_id: ID of the product to process.
        experiment_id: Experiment identifier; defaults to settings.EXPERIMENT_ID if None.
        settings: Engine Settings instance (provides LLM, Typesense, Compat config).
        metrics_repo: Injected MetricsRepository for writing results.
        http: Shared httpx.AsyncClient for Typesense requests (pooled, caller-owned).
        attr_vocab: Optional per-type attribute vocabulary from
            CatalogRepository.attribute_vocab_by_type(). Passed to _verify_jit
            so rule-gen prompts reference real attribute keys (vocab-grounded).
            None means no grounding - rules may reference invented keys (L=0).
        strategy: Retrieval strategy override; defaults to settings.RETRIEVAL_STRATEGY
            when None. Lets the caller (worker/experiment) select semantic vs cat_priors
            per run without restarting the engine.

    Returns:
        Dict with keys: product_id, status, candidates (optional), written (optional).
    """
    experiment_id = experiment_id or settings.EXPERIMENT_ID
    strategy = strategy or settings.RETRIEVAL_STRATEGY
    ts_base = settings.typesense_base_url
    ts_key = settings.TYPESENSE_API_KEY
    ts_col = settings.TYPESENSE_COLLECTION
    compat_mode = getattr(settings, "COMPAT_MODE", "jit")

    client = (
        None
        if settings.LLM_MODE == "mock"
        else AsyncOpenAI(
            api_key=settings.OPENAI_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or None,
        )
    )
    llm = get_llm(settings, client)

    # Open instrumentation run.
    ctx = RunContext(
        repo=metrics_repo,
        product_id=product_id,
        pipeline_version=_PIPELINE_VERSION,
        mode=compat_mode,
        experiment_id=experiment_id,
    )
    run_id = await ctx.open()

    try:
        # Stage: retrieval
        with ctx.stage("compatibility") as retrieval_stage:
            source = await _fetch_source(http, ts_base, ts_key, ts_col, product_id)
            if not source:
                await ctx.finish("completed")
                return {"product_id": product_id, "status": "no_source", "written": 0}

            raw = await retrieve_candidates(
                strategy,
                http,
                ts_base,
                ts_key,
                ts_col,
                source,
                settings.COMPAT_TOP_K,
            )
            # cat_priors: complement map already guarantees complementary leaf
            # categories, so do not drop on shared parent/top category (that would
            # discard real accessories like a phone case sharing the phone's parent).
            filtered = filter_candidates(
                source, raw, drop_same_category=(strategy != "cat_priors")
            )
            # tau_S gate only applies to similarity-based retrieval. For cat_priors
            # the curated complement map IS the relevance signal; the semantic score
            # is similarity-to-source, which is intentionally LOW for complementary
            # items (a phone is not similar to its case). Gating cat_priors by tau_S
            # would discard exactly the accessories we want, so we trust the map and
            # let the LLM verify judge compatibility.
            if strategy == "cat_priors":
                cands = filtered
            else:
                cands = [
                    c
                    for c in filtered
                    if float(c.get("_score", 0.0)) >= settings.COMPAT_TAU_S
                ]
            retrieval_stage.items_processed = len(cands)

        if not cands:
            await ctx.finish("completed")
            return {"product_id": product_id, "status": "no_candidates", "written": 0}

        recs: list[dict] = []
        eval_rows: list[dict] = []

        if compat_mode == "oneshot":
            # Control arm: legacy one-shot LLM verify path (unchanged).
            with ctx.stage("recommendation") as ver_stage:
                bs = settings.COMPAT_BATCH_SIZE
                for k in range(0, len(cands), bs):
                    chunk = cands[k : k + bs]
                    results, _ti, _to = await llm.verify(source, chunk)
                    ver_stage.llm_calls += 1
                    ver_stage.tokens_in += _ti
                    ver_stage.tokens_out += _to
                    sem_by_id = {
                        c.get("product_id"): float(c.get("_score", 0.0)) for c in chunk
                    }
                    for r in results:
                        cid = r.get("candidate_id")
                        if cid not in sem_by_id:
                            continue
                        s = sem_by_id[cid]
                        lg = float(r.get("logical_score", 0.0))
                        hybrid = compute_hybrid_score(s, lg, settings.COMPAT_ALPHA)
                        verdict = _decide_verdict(strategy, s, lg, settings)
                        ctx_code = r.get("context_code")
                        recs.append(
                            {
                                "experiment_id": experiment_id,
                                "product_id": product_id,
                                "recommended_id": cid,
                                "context_code": ctx_code,
                                "semantic": s,
                                "logical": lg,
                                "hybrid": hybrid,
                                "alpha": settings.COMPAT_ALPHA,
                                "verdict": verdict,
                            }
                        )
                        eval_rows.append(
                            {
                                "run_id": run_id,
                                "experiment_id": experiment_id,
                                "product_i": product_id,
                                "product_j": cid,
                                "context_code": ctx_code,
                                "semantic": s,
                                "logical": lg,
                                "hybrid": hybrid,
                                "alpha": settings.COMPAT_ALPHA,
                                "verdict": verdict,
                                "rules_evaluated": r.get("rules_evaluated", 0),
                                "rules_passed": r.get("rules_passed", 0),
                                "rules_failed": r.get("rules_failed", 0),
                                "rules_undefined": r.get("rules_undefined", 0),
                                "evidence_claims": r.get("evidence_claims", 0),
                                "hallucinated_claims": r.get("hallucinated_claims", 0),
                            }
                        )
                ver_stage.items_processed = len(cands)
        else:
            # JIT path (default).
            rule_cache = _build_rule_cache(settings, metrics_repo)
            sem_by_id = {
                c.get("product_id"): float(c.get("_score", 0.0)) for c in cands
            }

            with ctx.stage("recommendation") as ver_stage:
                jit_results = await _verify_jit(
                    source,
                    cands,
                    llm,
                    rule_cache,
                    settings,
                    ver_stage,
                    attr_vocab=attr_vocab,
                )

            for r in jit_results:
                cid = r["candidate_id"]
                s = sem_by_id.get(cid, 0.0)
                lg = float(r["logical_score"])
                hybrid = compute_hybrid_score(s, lg, settings.COMPAT_ALPHA)
                verdict = _decide_verdict(strategy, s, lg, settings)
                ctx_code = r.get("context_code")
                recs.append(
                    {
                        "experiment_id": experiment_id,
                        "product_id": product_id,
                        "recommended_id": cid,
                        "context_code": ctx_code,
                        "semantic": s,
                        "logical": lg,
                        "hybrid": hybrid,
                        "alpha": settings.COMPAT_ALPHA,
                        "verdict": verdict,
                    }
                )
                eval_rows.append(
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "product_i": product_id,
                        "product_j": cid,
                        "context_code": ctx_code,
                        "semantic": s,
                        "logical": lg,
                        "hybrid": hybrid,
                        "alpha": settings.COMPAT_ALPHA,
                        "verdict": verdict,
                        "rules_evaluated": r.get("rules_evaluated", 0),
                        "rules_passed": r.get("rules_passed", 0),
                        "rules_failed": r.get("rules_failed", 0),
                        "rules_undefined": r.get("rules_undefined", 0),
                        "evidence_claims": r.get("evidence_claims", 0),
                        "hallucinated_claims": r.get("hallucinated_claims", 0),
                    }
                )

        # Write all evaluations (every candidate pair, regardless of verdict).
        if eval_rows:
            await metrics_repo.write_evaluations(eval_rows)

        # Write recommendations (verdict=1 rows only - unchanged contract).
        verdict_recs = [r for r in recs if r["verdict"]]
        written = (
            await metrics_repo.write_recommendations(verdict_recs)
            if verdict_recs
            else 0
        )

        await ctx.finish("completed")
        return {
            "product_id": product_id,
            "status": "ok",
            "candidates": len(cands),
            "written": written,
        }

    except Exception as exc:
        error_msg = str(exc)[:500]
        try:
            await ctx.finish("failed", error=error_msg)
        except Exception as finish_exc:
            log.warning("failed to record run failure: %s", finish_exc)
        raise
