"""OpenAI Batch API pipeline for compatibility verification (-50% cost, async).

One request per product: ALL of the product's candidates go in a single chat
request (not split into batches of 10). Resume-aware: products already present in
the recommendations table are skipped.

Commands:
    python -m batch_run submit          [--chunk 5000] [--limit N]
    python -m batch_run status
    python -m batch_run collect
    python -m batch_run orchestrate     [--max-inflight 4] [--poll 120]
    python -m batch_run rulegen         [--limit N] [--dry-run]
    python -m batch_run rulegen-collect

The `rulegen` subcommand enumerates DISTINCT (source_type, cand_type) pairs from
candidate sets across products, submits ONE Batch API request per UNCACHED pair
(for rule-gen). This is the Batch API-accelerated companion to the online JIT path.

The `rulegen-collect` subcommand downloads completed rulegen batch output and writes
validated rule sets to the rule_cache table (and Redis when configured). It is
resumable and idempotent: re-run until it reports ALL_DONE. $0 - no new LLM calls.

--dry-run on rulegen prints pair count + token estimate with $0 spend.

Artifacts (host-visible via /src mount -> E:\\AVTC\\engine\\src\\):
    /docs/runs/batch_ids.json        list of {batch_id, input_file, n_requests}
    /docs/runs/batch_sidecar.json    custom_id -> {product_id, semantic:{cid:score}}
    /docs/runs/rulegen_ids.json      batch_ids for rulegen job
    /docs/runs/batch_input_*.jsonl   request payloads uploaded to OpenAI
    /docs/runs/batch_output_*.jsonl  downloaded results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

import httpx
from openai import AsyncOpenAI

from app.core.config import settings
from app.db.adapter import MySQLAdapter
from app.db.catalog_repository import CatalogRepository
from app.db.metrics_repository import MetricsRepository
from app.services.compatibility.cache import MockRuleCache, TableRuleCache
from app.services.compatibility.ontology import (
    RULEGEN_RESPONSE_FORMAT,
    RULEGEN_SYSTEM_PROMPT,
    MockRuleGen,
    build_rulegen_prompt,
    get_rules,
)
from app.services.llm.verifier import (
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    build_verification_prompt,
    parse_verification_response,
)
from app.services.retrieval.candidates import (
    filter_candidates,
    load_products,
    retrieve_candidates,
)
from app.services.scoring import compute_hybrid_score, compute_verdict

LOG_DIR = "/docs/runs"
IDS_FILE = f"{LOG_DIR}/batch_ids.json"
SIDECAR_FILE = f"{LOG_DIR}/batch_sidecar.json"
RULEGEN_IDS_FILE = f"{LOG_DIR}/rulegen_ids.json"
MAX_OUT_TOKENS = 4000  # plenty for one COMPAT_BATCH_SIZE (10) batch; avoids truncation
# Token estimate for one rulegen request: ~350 input (system+user) + ~500 output.
# Split: 70% input / 30% output of total 850 tokens.
_RULEGEN_TOKENS_PER_PAIR = 850
_RULEGEN_INPUT_FRAC = 0.70
_RULEGEN_OUTPUT_FRAC = 0.30
# gpt-5-nano Batch API rates (already the -50% batch rates):
#   Input:  $0.05 / 1M tokens
#   Output: $0.40 / 1M tokens
_RULEGEN_COST_PER_1M_INPUT = 0.05
_RULEGEN_COST_PER_1M_OUTPUT = 0.40


async def _make_metrics_repo() -> tuple[MySQLAdapter, MetricsRepository]:
    """Create and connect a metrics adapter + repository."""
    adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_METRICS_DATABASE,
    )
    await adapter.connect()
    return adapter, MetricsRepository(adapter)


async def _make_catalog_repo() -> tuple[MySQLAdapter, CatalogRepository]:
    """Create and connect a catalog adapter + repository."""
    adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    await adapter.connect()
    return adapter, CatalogRepository(adapter)


# ---------------------------------------------------------------- submit -----
async def cmd_submit(chunk: int, limit: int):
    os.makedirs(LOG_DIR, exist_ok=True)
    http = httpx.AsyncClient(timeout=120.0)
    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    metrics_adapter, metrics_repo = await _make_metrics_repo()

    products = await load_products(
        http,
        settings.typesense_base_url,
        settings.TYPESENSE_API_KEY,
        settings.TYPESENSE_COLLECTION,
        limit=0,
    )
    done = await metrics_repo.done_product_ids(settings.EXPERIMENT_ID)
    products = [
        p for p in products if p["product_id"] not in done and (p.get("embedding"))
    ]
    if limit:
        products = products[:limit]
    print(f"to process: {len(products)} (skipped {len(done)} already done)")

    # fetch candidates concurrently (Typesense)
    sem = asyncio.Semaphore(16)
    sidecar: dict = {}
    requests: list[dict] = []

    async def build_one(p):
        """Return a LIST of requests for one product."""
        async with sem:
            raw = await retrieve_candidates(
                settings.RETRIEVAL_STRATEGY,
                http,
                settings.typesense_base_url,
                settings.TYPESENSE_API_KEY,
                settings.TYPESENSE_COLLECTION,
                p,
                settings.COMPAT_TOP_K,
            )
        cands = filter_candidates(p, raw)
        cands = [
            c for c in cands if float(c.get("_score", 0.0)) >= settings.COMPAT_TAU_S
        ]
        if not cands:
            return []
        reqs = []
        bs = settings.COMPAT_BATCH_SIZE
        for k in range(0, len(cands), bs):
            ch = cands[k : k + bs]
            cid = f"p{p['product_id']}b{k // bs}"
            sidecar[cid] = {
                "product_id": p["product_id"],
                "semantic": {
                    str(c["product_id"]): float(c.get("_score", 0.0)) for c in ch
                },
            }
            body = {
                "model": settings.PRIMARY_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_verification_prompt(p, ch)},
                ],
                "response_format": RESPONSE_FORMAT,
                "max_completion_tokens": MAX_OUT_TOKENS,
            }
            if settings.REASONING_EFFORT:
                body["reasoning_effort"] = settings.REASONING_EFFORT
            reqs.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )
        return reqs

    built = await asyncio.gather(*[build_one(p) for p in products])
    requests = [r for sub in built for r in sub]
    print(f"built {len(requests)} requests from {sum(1 for s in built if s)} products")

    with open(SIDECAR_FILE, "w", encoding="utf-8") as f:
        json.dump(sidecar, f)

    # chunk -> file -> upload -> create batch
    ids = []
    for ci in range(0, len(requests), chunk):
        part = requests[ci : ci + chunk]
        path = f"{LOG_DIR}/batch_input_{ci // chunk}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(path, "rb") as f:
            up = await client.files.create(file=f, purpose="batch")
        batch = await client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        ids.append({"batch_id": batch.id, "input_file": path, "n_requests": len(part)})
        print(f"submitted chunk {ci // chunk}: batch={batch.id} requests={len(part)}")

    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump({"experiment_id": settings.EXPERIMENT_ID, "chunks": ids}, f, indent=2)
    print(
        f"saved {len(ids)} batch id(s) for experiment={settings.EXPERIMENT_ID} -> {IDS_FILE}"
    )

    await http.aclose()
    await metrics_adapter.close()


def _load_ids() -> tuple[str, list]:
    """Load (experiment_id, chunks) from IDS_FILE. Tolerates the legacy bare-list format."""
    data = json.load(open(IDS_FILE, encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("experiment_id", "baseline_v1"), data.get("chunks", [])
    return "baseline_v1", data  # legacy: bare list of chunk dicts


def _save_ids(experiment_id: str, chunks: list) -> None:
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump({"experiment_id": experiment_id, "chunks": chunks}, f, indent=2)


# ---------------------------------------------------------------- status -----
async def cmd_status():
    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    exp, chunks = _load_ids()
    print(f"experiment={exp}")
    for item in chunks:
        b = await client.batches.retrieve(item["batch_id"])
        rc = b.request_counts
        print(
            f"{item['batch_id']}  status={b.status}  "
            f"total={rc.total} completed={rc.completed} failed={rc.failed}  "
            f"out_file={b.output_file_id}"
        )


def _collected_path() -> str:
    return f"{LOG_DIR}/collected.json"


def _load_collected() -> set:
    try:
        return set(json.load(open(_collected_path(), encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_collected(s: set) -> None:
    with open(_collected_path(), "w", encoding="utf-8") as f:
        json.dump(sorted(s), f)


# --------------------------------------------------------------- collect -----
async def cmd_collect():
    """Resumable, fault-tolerant collection of completed batches. Skips already-collected
    batch_ids (markers in collected.json), wraps each batch in try/except, and is upsert-
    idempotent. Re-run until it reports ALL_DONE. Makes no LLM calls (free)."""
    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    metrics_adapter, metrics_repo = await _make_metrics_repo()
    sidecar = json.load(open(SIDECAR_FILE, encoding="utf-8"))
    exp, chunks = _load_ids()
    collected = _load_collected()

    pending = 0
    try:
        for item in chunks:
            bid = item["batch_id"]
            if bid in collected:
                continue
            try:
                b = await client.batches.retrieve(bid)
                if b.status != "completed" or not b.output_file_id:
                    print(f"{bid}: status={b.status}, not ready", flush=True)
                    pending += 1
                    continue
                n = await _collect_one(client, b, sidecar, metrics_repo, exp)
                collected.add(bid)
                _save_collected(collected)
                print(
                    f"{bid}: wrote {n} recommendations "
                    f"[collected {len(collected)}/{len(chunks)}]",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"{bid}: error, will retry on next run ({str(e)[:80]})", flush=True
                )
                pending += 1
        remaining = len(chunks) - len(collected)
        print(
            f"{'ALL_DONE' if remaining == 0 else 'PARTIAL'}: "
            f"collected={len(collected)}/{len(chunks)} pending={pending}",
            flush=True,
        )
    finally:
        await metrics_adapter.close()


async def _collect_one(
    client, b, sidecar, metrics_repo: MetricsRepository, experiment_id
) -> int:
    """Download one completed batch's output, score, write. Returns count written."""
    if not b.output_file_id:
        return 0
    content = await client.files.content(b.output_file_id)
    out_path = f"{LOG_DIR}/batch_output_{b.id}.jsonl"
    with open(out_path, "wb") as f:
        f.write(content.read())

    recs = []
    for line in open(out_path, encoding="utf-8"):
        if not line.strip():
            continue
        obj = json.loads(line)
        meta = sidecar.get(obj.get("custom_id"))
        if not meta:
            continue
        resp = (obj.get("response") or {}).get("body") or {}
        choices = resp.get("choices") or []
        if not choices:
            continue
        raw = choices[0].get("message", {}).get("content") or ""
        sem_map = meta["semantic"]
        for r in parse_verification_response(raw):
            s = sem_map.get(str(r.get("candidate_id")))
            if s is None:
                continue
            logical = float(r.get("logical_score", 0.0))
            recs.append(
                {
                    "experiment_id": experiment_id,
                    "product_id": meta["product_id"],
                    "recommended_id": r.get("candidate_id"),
                    "context_code": r.get("context_code"),
                    "semantic": s,
                    "logical": logical,
                    "hybrid": compute_hybrid_score(s, logical, settings.COMPAT_ALPHA),
                    "alpha": settings.COMPAT_ALPHA,
                    "verdict": compute_verdict(
                        s, logical, settings.COMPAT_TAU_S, settings.COMPAT_TAU_L
                    ),
                }
            )
    if recs:
        await metrics_repo.write_recommendations(recs)
    return len(recs)


async def cmd_orchestrate(max_inflight: int, poll: int):
    """Self-feeding loop: keep <=max_inflight batches enqueued (20M-token limit),
    resubmit failed/pending chunks, collect completed ones, until all done."""
    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    metrics_adapter, metrics_repo = await _make_metrics_repo()
    sidecar = json.load(open(SIDECAR_FILE, encoding="utf-8"))
    experiment_id, chunks = _load_ids()
    print(f"orchestrating experiment={experiment_id}, {len(chunks)} chunks", flush=True)
    done = set()

    ACTIVE = {"validating", "in_progress", "finalizing"}
    DEAD = {"failed", "expired", "cancelled", "cancelling"}

    while len(done) < len(chunks):
        active = 0
        for i, ch in enumerate(chunks):
            if i in done:
                continue
            bid = ch.get("batch_id")
            if bid:
                try:
                    b = await client.batches.retrieve(bid)
                    if b.status == "completed":
                        n = await _collect_one(
                            client, b, sidecar, metrics_repo, experiment_id
                        )
                        print(f"chunk {i}: completed, wrote {n}", flush=True)
                        done.add(i)
                        continue
                    if b.status in ACTIVE:
                        active += 1
                        continue
                    if b.status in DEAD:
                        ch["batch_id"] = None  # needs resubmit
                except Exception as e:
                    print(
                        f"chunk {i}: retrieve/collect error, retry next tick ({str(e)[:80]})",
                        flush=True,
                    )
                    active += 1
                    continue

        # feed pending chunks up to the inflight cap
        for i, ch in enumerate(chunks):
            if i in done or active >= max_inflight or ch.get("batch_id"):
                continue
            try:
                with open(ch["input_file"], "rb") as f:
                    up = await client.files.create(file=f, purpose="batch")
                b = await client.batches.create(
                    input_file_id=up.id,
                    endpoint="/v1/chat/completions",
                    completion_window="24h",
                )
                ch["batch_id"] = b.id
                active += 1
                print(f"chunk {i}: (re)submitted {b.id}", flush=True)
            except Exception as e:
                print(f"chunk {i}: submit deferred ({str(e)[:80]})", flush=True)

        _save_ids(experiment_id, chunks)
        print(f"progress: done={len(done)}/{len(chunks)} active={active}", flush=True)
        if len(done) >= len(chunks):
            break
        await asyncio.sleep(poll)

    print("ORCHESTRATE DONE: all chunks collected", flush=True)
    await metrics_adapter.close()


async def _collect_distinct_pairs(
    http: httpx.AsyncClient,
    limit: int,
) -> list[tuple[str, str]]:
    """Enumerate DISTINCT (source_type, cand_type) pairs from Typesense candidate sets.

    Loads up to `limit` products (0 = all), fetches each product's candidate set,
    and returns the unique type-pair set as a sorted list (deterministic order).

    Args:
        http: Shared httpx.AsyncClient.
        limit: Maximum number of products to scan (0 = all).

    Returns:
        Sorted list of distinct (source_type, cand_type) tuples.
    """
    products = await load_products(
        http,
        settings.typesense_base_url,
        settings.TYPESENSE_API_KEY,
        settings.TYPESENSE_COLLECTION,
        limit=0,
    )
    if limit:
        products = products[:limit]

    sem = asyncio.Semaphore(16)
    pairs: set[tuple[str, str]] = set()

    async def scan_one(p):
        async with sem:
            raw = await retrieve_candidates(
                settings.RETRIEVAL_STRATEGY,
                http,
                settings.typesense_base_url,
                settings.TYPESENSE_API_KEY,
                settings.TYPESENSE_COLLECTION,
                p,
                settings.COMPAT_TOP_K,
            )
        cands = filter_candidates(p, raw)
        cands = [
            c for c in cands if float(c.get("_score", 0.0)) >= settings.COMPAT_TAU_S
        ]
        src_type = str(p.get("product_type") or "unknown")
        for c in cands:
            ctype = str(c.get("product_type") or "unknown")
            pairs.add((src_type, ctype))

    await asyncio.gather(*[scan_one(p) for p in products])
    return sorted(pairs)


async def cmd_rulegen(limit: int, dry_run: bool) -> None:
    """Enumerate distinct type-pairs and submit Batch API requests for rule-gen.

    In mock mode (LLM_MODE=mock) or dry_run mode: prints pair count and token
    estimate only - no Batch API calls, $0 spend.

    In real mode (LLM_MODE=real, not dry_run): submits one Batch API request per
    UNCACHED type-pair, saves rulegen_ids.json. On a subsequent `rulegen collect`
    call, parsed rule sets are written to rule_cache table.

    Args:
        limit: Limit product scan to N products (0 = all).
        dry_run: Print estimate only, no API calls.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    http = httpx.AsyncClient(timeout=120.0)
    metrics_adapter, metrics_repo = await _make_metrics_repo()
    catalog_adapter, catalog_repo = await _make_catalog_repo()

    try:
        print("scanning candidate sets for distinct type-pairs...")
        all_pairs = await _collect_distinct_pairs(http, limit)
        print(f"found {len(all_pairs)} distinct type-pairs total")

        # Filter out already-cached pairs.
        if settings.LLM_MODE != "mock":
            table_cache = TableRuleCache(metrics_repo)
            uncached: list[tuple[str, str]] = []
            for type_a, type_b in all_pairs:
                cached = await table_cache.get(type_a, type_b)
                if cached is None:
                    uncached.append((type_a, type_b))
            print(
                f"uncached pairs: {len(uncached)} "
                f"(cached: {len(all_pairs) - len(uncached)})"
            )
        else:
            uncached = all_pairs

        # Build per-type attribute vocabulary to ground rule-gen prompts.
        # Rules that reference real keys are evaluable; invented keys produce
        # rules_undefined=N, L=0, zero recommendations.
        print("building attribute vocabulary from product_ai_data...")
        vocab = await catalog_repo.attribute_vocab_by_type(top_n=25)
        print(f"vocab built for {len(vocab)} product types")

        token_estimate = len(uncached) * _RULEGEN_TOKENS_PER_PAIR
        input_tokens = int(token_estimate * _RULEGEN_INPUT_FRAC)
        output_tokens = int(token_estimate * _RULEGEN_OUTPUT_FRAC)
        input_cost = input_tokens * _RULEGEN_COST_PER_1M_INPUT / 1_000_000
        output_cost = output_tokens * _RULEGEN_COST_PER_1M_OUTPUT / 1_000_000
        total_cost = input_cost + output_cost
        print(f"token estimate for {len(uncached)} pairs: ~{token_estimate:,} tokens")
        print(f"  input  tokens: ~{input_tokens:,} @ ${_RULEGEN_COST_PER_1M_INPUT}/1M")
        print(
            f"  output tokens: ~{output_tokens:,} @ ${_RULEGEN_COST_PER_1M_OUTPUT}/1M"
        )
        print(
            f"estimated cost (gpt-5-nano batch, $0.05/1M in, $0.40/1M out): "
            f"~${total_cost:.4f}"
        )
        print(
            "note: 16,605 pairs is on the current (granular) product_type; "
            "coarse normalized types will reduce this."
        )

        if dry_run:
            print("[dry-run] no Batch API calls submitted")
            for ta, tb in uncached[:20]:
                va = vocab.get(ta)
                vb = vocab.get(tb)
                print(
                    f"  pair: {ta} -> {tb} "
                    f"(vocab_a={len(va) if va else 0} keys, "
                    f"vocab_b={len(vb) if vb else 0} keys)"
                )
            if len(uncached) > 20:
                print(f"  ... and {len(uncached) - 20} more")
            return

        if settings.LLM_MODE == "mock":
            print("[mock mode] no real Batch API calls - using MockRuleGen inline")
            cache = MockRuleCache()
            llm = MockRuleGen()
            for type_a, type_b in uncached:
                rules, _, _, _ = await get_rules(
                    type_a, type_b, llm, cache, hdr_enabled=False
                )
                print(f"  mock generated {len(rules)} rules for {type_a} -> {type_b}")
            return

        # Real mode: build JSONL, upload, submit batch.
        client = AsyncOpenAI(
            api_key=settings.OPENAI_KEY.get_secret_value(),
            base_url=settings.OPENAI_BASE_URL or None,
        )
        requests: list[dict] = []
        for type_a, type_b in uncached:
            user_prompt = build_rulegen_prompt(
                type_a,
                type_b,
                vocab_a=vocab.get(type_a),
                vocab_b=vocab.get(type_b),
            )
            cid = f"rulegen_{type_a}__{type_b}"
            body = {
                "model": settings.PRIMARY_MODEL,
                "messages": [
                    {"role": "system", "content": RULEGEN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": RULEGEN_RESPONSE_FORMAT,
                "max_completion_tokens": settings.MAX_TOKENS,
            }
            if settings.REASONING_EFFORT:
                body["reasoning_effort"] = settings.REASONING_EFFORT
            requests.append(
                {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body,
                }
            )

        path = f"{LOG_DIR}/rulegen_input.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in requests:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        with open(path, "rb") as f:
            up = await client.files.create(file=f, purpose="batch")
        batch = await client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        ids_data = {
            "batch_id": batch.id,
            "input_file": path,
            "n_requests": len(requests),
            "pairs": [[ta, tb] for ta, tb in uncached],
        }
        with open(RULEGEN_IDS_FILE, "w", encoding="utf-8") as f:
            json.dump(ids_data, f, indent=2)
        print(
            f"submitted rulegen batch: batch_id={batch.id} pairs={len(requests)} -> {RULEGEN_IDS_FILE}"
        )
    finally:
        await http.aclose()
        await metrics_adapter.close()
        await catalog_adapter.close()


def _rulegen_collected_path() -> str:
    return f"{LOG_DIR}/rulegen_collected.json"


def _load_rulegen_collected() -> set:
    try:
        return set(json.load(open(_rulegen_collected_path(), encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return set()


def _save_rulegen_collected(s: set) -> None:
    with open(_rulegen_collected_path(), "w", encoding="utf-8") as f:
        json.dump(sorted(s), f)


def _parse_rulegen_line(line: str) -> tuple[str, str, list[dict]] | None:
    """Parse one JSONL line from a rulegen batch output file.

    Extracts type_a and type_b from the custom_id (format: ``rulegen_{a}__{b}``),
    and parses the rule list from choices[0].message.content. Tolerates markdown
    fences and truncated/malformed JSON (returns None on any parse failure).

    This is a pure function with no I/O - it can be unit-tested directly.

    Args:
        line: One raw JSONL line from a batch output file.

    Returns:
        Tuple of (type_a, type_b, raw_rules_list) on success, or None on any
        parse failure (missing fields, bad JSON, missing prefix/separator).
    """
    import logging as _logging

    _log = _logging.getLogger(__name__)

    if not line.strip():
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        _log.warning("rulegen-collect: could not parse JSONL line")
        return None

    custom_id = obj.get("custom_id", "")
    if not custom_id.startswith("rulegen_"):
        return None

    # Strip the "rulegen_" prefix then split on the first "__"
    remainder = custom_id[len("rulegen_") :]
    if "__" not in remainder:
        _log.warning("rulegen-collect: malformed custom_id (no __): %s", custom_id)
        return None
    type_a, type_b = remainder.split("__", 1)
    if not type_a or not type_b:
        _log.warning("rulegen-collect: empty type in custom_id: %s", custom_id)
        return None

    resp = (obj.get("response") or {}).get("body") or {}
    choices = resp.get("choices") or []
    if not choices:
        _log.warning("rulegen-collect: no choices in response for %s", custom_id)
        return None

    raw_content = choices[0].get("message", {}).get("content") or ""
    # Tolerate markdown fences (same as parse_results in verifier)
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        _log.warning(
            "rulegen-collect: bad JSON in content for %s: %.160s", custom_id, text
        )
        return None

    if isinstance(data, dict):
        raw_rules = data.get("rules", [])
    elif isinstance(data, list):
        raw_rules = data
    else:
        raw_rules = []

    if not isinstance(raw_rules, list):
        _log.warning("rulegen-collect: rules is not a list for %s", custom_id)
        return None

    return type_a, type_b, raw_rules


# ---------------------------------------------------------- rulegen-collect ---


async def cmd_rulegen_collect() -> None:
    """Download completed rulegen batch output and write rules to rule_cache.

    Reads RULEGEN_IDS_FILE for the batch_id. Retrieves the batch via the OpenAI
    client. If the batch is not yet completed (status != 'completed' or no
    output_file_id), prints "not ready" and exits cleanly so the caller can retry.

    For each JSONL line in the output:
    - Parses custom_id: ``rulegen_{type_a}__{type_b}``
    - Extracts rule JSON from choices[0].message.content (tolerant parse)
    - Validates via _validate_rules (drops malformed; skips pair if < 2 valid rules)
    - Upserts to rule_cache via metrics_repo.set_rule_cache (idempotent)

    Resumable: already-collected batch_ids are recorded in rulegen_collected.json.
    set_rule_cache uses INSERT ... ON DUPLICATE KEY UPDATE so re-runs are safe.

    Re-run until output reports ALL_DONE. $0 - no new LLM generation.
    """
    from app.services.compatibility.ontology import _validate_rules

    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    metrics_adapter, metrics_repo = await _make_metrics_repo()

    try:
        if not os.path.exists(RULEGEN_IDS_FILE):
            print(f"no rulegen ids file found at {RULEGEN_IDS_FILE}", flush=True)
            print("run 'rulegen' subcommand first to submit a batch", flush=True)
            return

        ids_data = json.load(open(RULEGEN_IDS_FILE, encoding="utf-8"))
        # ids_data may be a single dict (one batch) or a list of dicts
        if isinstance(ids_data, dict):
            batch_entries = [ids_data]
        else:
            batch_entries = list(ids_data)

        collected = _load_rulegen_collected()
        pending = 0
        total_written = 0
        total_skipped = 0

        for entry in batch_entries:
            bid = entry.get("batch_id")
            if not bid:
                continue
            if bid in collected:
                print(f"{bid}: already collected, skipping", flush=True)
                continue

            try:
                b = await client.batches.retrieve(bid)
            except Exception as e:
                print(
                    f"{bid}: retrieve error, will retry on next run ({str(e)[:80]})",
                    flush=True,
                )
                pending += 1
                continue

            if b.status != "completed" or not b.output_file_id:
                print(f"{bid}: status={b.status}, not ready", flush=True)
                pending += 1
                continue

            # Download output JSONL
            try:
                content = await client.files.content(b.output_file_id)
                out_path = f"{LOG_DIR}/rulegen_output_{bid}.jsonl"
                os.makedirs(LOG_DIR, exist_ok=True)
                with open(out_path, "wb") as fh:
                    fh.write(content.read())
            except Exception as e:
                print(
                    f"{bid}: download error, will retry on next run ({str(e)[:80]})",
                    flush=True,
                )
                pending += 1
                continue

            written = 0
            skipped = 0
            for line in open(out_path, encoding="utf-8"):
                parsed = _parse_rulegen_line(line)
                if parsed is None:
                    continue
                type_a, type_b, raw_rules = parsed
                try:
                    rules = _validate_rules(raw_rules)
                except ValueError as ve:
                    print(f"  {type_a} -> {type_b}: skipped ({ve})", flush=True)
                    skipped += 1
                    continue
                try:
                    await metrics_repo.set_rule_cache(
                        type_a,
                        type_b,
                        rules,
                        generated_by="batch_rulegen",
                        source_hash="",
                    )
                    written += 1
                except Exception as e:
                    print(
                        f"  {type_a} -> {type_b}: write error ({str(e)[:80]})",
                        flush=True,
                    )
                    skipped += 1

            collected.add(bid)
            _save_rulegen_collected(collected)
            total_written += written
            total_skipped += skipped
            print(
                f"{bid}: wrote {written} rule sets, skipped {skipped} pairs "
                f"[collected {len(collected)}/{len(batch_entries)}]",
                flush=True,
            )

        remaining = len(batch_entries) - len(collected)
        print(
            f"{'ALL_DONE' if remaining == 0 and pending == 0 else 'PARTIAL'}: "
            f"pairs_written={total_written} pairs_skipped={total_skipped} "
            f"batches_pending={pending}",
            flush=True,
        )
    finally:
        await metrics_adapter.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--chunk", type=int, default=5000)
    s.add_argument("--limit", type=int, default=0)
    sub.add_parser("status")
    sub.add_parser("collect")
    o = sub.add_parser("orchestrate")
    o.add_argument("--max-inflight", type=int, default=4)
    o.add_argument("--poll", type=int, default=120)
    rg = sub.add_parser("rulegen")
    rg.add_argument("--limit", type=int, default=0, help="limit product scan (0=all)")
    rg.add_argument(
        "--dry-run", action="store_true", help="print estimate, no API calls"
    )
    sub.add_parser(
        "rulegen-collect",
        help="collect completed rulegen batch output into rule_cache table",
    )
    args = ap.parse_args()

    if args.cmd == "submit":
        asyncio.run(cmd_submit(args.chunk, args.limit))
    elif args.cmd == "status":
        asyncio.run(cmd_status())
    elif args.cmd == "collect":
        asyncio.run(cmd_collect())
    elif args.cmd == "orchestrate":
        asyncio.run(cmd_orchestrate(args.max_inflight, args.poll))
    elif args.cmd == "rulegen":
        asyncio.run(cmd_rulegen(args.limit, args.dry_run))
    elif args.cmd == "rulegen-collect":
        asyncio.run(cmd_rulegen_collect())


if __name__ == "__main__":
    main()
