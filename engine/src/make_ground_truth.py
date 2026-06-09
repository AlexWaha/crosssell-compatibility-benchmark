"""CLI: build a reproducible, independent ground-truth via the LLM judge.

The ground truth is created automatically (no manual labelling). For each anchor
product (a product in an anchor category) the FIXED candidate universe is every
product sitting in one of that anchor's complementary categories (from the
category complement map). A strong judge model (default JUDGE_MODEL=gpt-5) reads
the REAL product data and labels each pair 1/0 with a rationale.

Why this design:
  - Independent of the retrieval/verify pipeline: the candidate universe depends
    only on (dataset, complement map), NOT on top_k / tau / strategy. Any
    experiment config is therefore scored against the SAME stable ground truth.
  - Reproducible: deterministic ordering, INSERT IGNORE caching (reruns judge
    only the not-yet-labelled pairs), fixed model + prompt.
  - Catches model-level compatibility (e.g. an iPhone has no microSD slot) that a
    coarse category-only graph cannot.

Usage (run inside the engine container):
    # $0 dry-run: print universe size + projected cost, write nothing
    python -m make_ground_truth --dry-run

    # calibrate real cost on a few anchors
    python -m make_ground_truth --authorize --limit-anchors 5

    # full reproducible build (real judge, budget-guarded)
    python -m make_ground_truth --authorize --max-cost 12

Safety: real LLM is used ONLY with --authorize AND LLM_MODE=real. A --max-cost
guard (USD) stops the run before the budget is exceeded.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os

import aiomysql
import httpx

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.adapter import MySQLAdapter
from app.db.metrics_repository import MetricsRepository
from app.services.compatibility.judge import judge_batch
from app.services.llm.verifier import get_llm

log = logging.getLogger(__name__)

# Anchor categories (focused verticals). Products in these categories are the
# sources whose complementary universe gets judged. Override with --anchor-cats.
DEFAULT_ANCHOR_CATS = [623, 246, 1730, 1390, 373, 152, 15, 161, 749, 258, 2176, 882]

_COMPLEMENT_MAP_PATH = os.path.join(
    os.path.dirname(__file__),
    "app",
    "services",
    "retrieval",
    "category_complements.json",
)

# USD per 1M tokens (input, output). Used only for the budget guard / projection.
PRICES = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
}


def _price(model: str) -> tuple[float, float]:
    # Match the MOST specific prefix first ('gpt-5-mini'/'gpt-5-nano' before 'gpt-5'),
    # otherwise 'gpt-5' would shadow its cheaper variants and overstate cost.
    for key in sorted(PRICES, key=len, reverse=True):
        if model.startswith(key):
            return PRICES[key]
    return (1.25, 10.0)


async def _load_universe(
    anchor_cats: list[int],
) -> tuple[dict[int, set[int]], dict[int, set[str]]]:
    """Build {anchor_id -> set(candidate_ids)} from catalog + complement map.

    Deterministic: derived only from product_categories and the complement map.
    Returns (universe, prod_catnames) where prod_catnames maps product->category
    names (used to attach context).
    """
    cmap = json.loads(open(_COMPLEMENT_MAP_PATH, encoding="utf-8").read())
    conn = await aiomysql.connect(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_CATALOG_DATABASE,
    )
    try:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT pc.product_id, c.category_id, c.name "
                "FROM product_categories pc JOIN categories c "
                "ON c.category_id = pc.category_id"
            )
            rows = await cur.fetchall()
    finally:
        conn.close()

    prod_catids: dict[int, set[int]] = collections.defaultdict(set)
    prod_catnames: dict[int, set[str]] = collections.defaultdict(set)
    catname_prods: dict[str, set[int]] = collections.defaultdict(set)
    for pid, cid, cname in rows:
        prod_catids[pid].add(cid)
        prod_catnames[pid].add(cname)
        catname_prods[cname].add(pid)

    anchor_set = set(anchor_cats)
    universe: dict[int, set[int]] = {}
    for pid, cids in prod_catids.items():
        if not (cids & anchor_set):
            continue
        comp_cats: set[str] = set()
        for cname in prod_catnames[pid]:
            comp_cats.update(cmap.get(cname, []))
        cand: set[int] = set()
        for cc in comp_cats:
            cand |= catname_prods.get(cc, set())
        cand.discard(pid)
        if cand:
            universe[pid] = cand
    return universe, prod_catnames


async def _fetch_docs(http: httpx.AsyncClient, ids: list[int]) -> dict[int, dict]:
    """Fetch product docs (name, product_type, attributes) from Typesense by id."""
    base = settings.typesense_base_url
    col = settings.TYPESENSE_COLLECTION
    out: dict[int, dict] = {}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        flt = "product_id:=[" + ",".join(str(x) for x in chunk) + "]"
        resp = await http.get(
            f"{base}/collections/{col}/documents/search",
            params={
                "q": "*",
                "query_by": "name",
                "filter_by": flt,
                "per_page": len(chunk),
                "exclude_fields": "embedding",
            },
            headers={"X-TYPESENSE-API-KEY": settings.TYPESENSE_API_KEY},
        )
        resp.raise_for_status()
        for hit in resp.json().get("hits", []):
            doc = hit["document"]
            raw = doc.get("attributes_json")
            attrs = {}
            if raw:
                try:
                    attrs = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    attrs = {}
            out[int(doc["product_id"])] = {
                "product_id": int(doc["product_id"]),
                "name": doc.get("name"),
                "product_type": doc.get("product_type"),
                "categories": doc.get("categories") or [],
                "attributes": attrs,
            }
    return out


async def main(args: argparse.Namespace) -> None:
    setup_logging(settings.LOG_LEVEL)
    anchor_cats = (
        [int(x) for x in args.anchor_cats.split(",")]
        if args.anchor_cats
        else DEFAULT_ANCHOR_CATS
    )
    universe, _ = await _load_universe(anchor_cats)
    anchors = sorted(universe)
    if args.limit_anchors:
        anchors = anchors[: args.limit_anchors]
    total_pairs = sum(len(universe[a]) for a in anchors)
    p_in, p_out = _price(settings.JUDGE_MODEL)
    est_calls = sum(
        (len(universe[a]) + args.batch - 1) // args.batch for a in anchors
    )
    est_cost = (est_calls * 700) / 1e6 * p_in + (est_calls * 350) / 1e6 * p_out
    print(
        f"anchors={len(anchors)} pairs={total_pairs} batch={args.batch} "
        f"est_calls={est_calls} model={settings.JUDGE_MODEL} "
        f"est_cost=${est_cost:.2f} max_cost=${args.max_cost}"
    )

    use_real = args.authorize and settings.LLM_MODE == "real"
    if args.dry_run:
        print("[dry-run] no rows written")
        return
    if not use_real:
        print("ERROR: real judge needs --authorize AND LLM_MODE=real. Aborting.")
        return

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=settings.OPENAI_KEY.get_secret_value(),
        base_url=settings.OPENAI_BASE_URL or None,
    )
    llm = get_llm(settings, client)

    metrics_adapter = MySQLAdapter(
        host=settings.DB_HOST,
        port=settings.MYSQL_PORT,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD.get_secret_value(),
        db=settings.DB_METRICS_DATABASE,
    )
    await metrics_adapter.connect()
    repo = MetricsRepository(metrics_adapter)

    # Already-judged pairs (resume / idempotency).
    async with metrics_adapter.cursor(dict_cursor=False) as cur:
        await cur.execute(
            "SELECT product_i, product_j FROM ground_truth WHERE source='llm'"
        )
        done = {(int(a), int(b)) for a, b in await cur.fetchall()}

    spent = 0.0
    written = 0
    skipped = 0
    tok_in = tok_out = 0
    try:
        sem = asyncio.Semaphore(args.concurrency)
        stop = asyncio.Event()
        state = {"written": 0, "skipped": 0, "tok_in": 0, "tok_out": 0, "spent": 0.0}

        async def judge_one_anchor(ai: int, anchor: int) -> None:
            cands = sorted(c for c in universe[anchor] if (anchor, c) not in done)
            if not cands:
                return
            docs = await _fetch_docs(http, [anchor] + cands)
            src = docs.get(anchor)
            if not src:
                return
            for k in range(0, len(cands), args.batch):
                if stop.is_set():
                    return
                chunk_docs = [docs[c] for c in cands[k : k + args.batch] if c in docs]
                if not chunk_docs:
                    continue
                try:
                    rows, ti, to = await judge_batch(
                        src, chunk_docs, llm, settings.JUDGE_MODEL
                    )
                except Exception as exc:
                    # A single bad batch (moderation flag, transient API error)
                    # must not abort the whole reproducible run.
                    state["skipped"] += len(chunk_docs)
                    log.warning(
                        "judge batch skipped anchor=%s n=%d: %.140s",
                        anchor,
                        len(chunk_docs),
                        str(exc),
                    )
                    continue
                state["tok_in"] += ti
                state["tok_out"] += to
                state["spent"] = (
                    state["tok_in"] / 1e6 * p_in + state["tok_out"] / 1e6 * p_out
                )
                if rows:
                    state["written"] += await repo.write_ground_truth(rows)
                if state["spent"] >= args.max_cost:
                    stop.set()
                    print(f"budget guard hit at ${state['spent']:.2f} - stopping")
            if ai % 20 == 0:
                print(
                    f"  anchor {ai}/{len(anchors)} written={state['written']} "
                    f"spent=${state['spent']:.2f}"
                )

        async def safe_anchor(ai: int, a: int) -> None:
            # Anchor-level semaphore: bound how many anchors are in flight at once
            # (each anchor fetches ~150 docs + judges). Without this, gather() starts
            # all anchors simultaneously -> hundreds of concurrent Typesense fetches
            # and a huge doc set in memory -> the process gets OOM-killed mid-run.
            async with sem:
                try:
                    await judge_one_anchor(ai, a)
                except Exception as exc:
                    log.warning("anchor %s failed (skipped): %.140s", a, str(exc))

        async with httpx.AsyncClient(timeout=120.0) as http:
            await asyncio.gather(
                *(safe_anchor(ai, a) for ai, a in enumerate(anchors, 1)),
                return_exceptions=True,
            )
        written = state["written"]
        skipped = state["skipped"]
        tok_in = state["tok_in"]
        tok_out = state["tok_out"]
        spent = state["spent"]
    finally:
        await metrics_adapter.close()
    print(
        f"DONE: written={written} rows, skipped={skipped}, "
        f"tokens_in={tok_in} tokens_out={tok_out} cost=${spent:.2f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--anchor-cats", default="", help="CSV of anchor category ids")
    ap.add_argument("--batch", type=int, default=15, help="candidates per judge call")
    ap.add_argument("--limit-anchors", type=int, default=0, help="cap anchors (calibration)")
    ap.add_argument("--max-cost", type=float, default=12.0, help="USD budget guard")
    ap.add_argument("--concurrency", type=int, default=8, help="concurrent judge calls")
    ap.add_argument("--authorize", action="store_true", help="enable real judge")
    ap.add_argument("--dry-run", action="store_true", help="size + cost only")
    asyncio.run(main(ap.parse_args()))
