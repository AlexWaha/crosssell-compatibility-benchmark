# Reproducing the AVTC cross-sell experiment

Everything below is deterministic and automated. Ground truth is created by an LLM
judge, never by hand. Two commands reproduce the full result.

```bash
# click 1 - data + ground truth (idempotent; safe to re-run)
./benchmark/reproduce.sh data

# click 2 - run the experiment and print metrics
./benchmark/reproduce.sh run
```

`./benchmark/reproduce.sh all` does both in sequence.

## What each stage does

| Stage | Command (inside engine) | Output | Deterministic via |
|-------|-------------------------|--------|-------------------|
| 1 build dataset | `python benchmark/build.py` | `benchmark/*.json` (3000 products) | `SEED=42` |
| 2 import catalog | `python -m catalog_importer` | `avtc_catalog` tables | dataset is source of truth |
| 3 normalize | `python -m normalize` | `product_ai_data` | `source_hash` skip |
| 4 embed | `python -m index_products run` | Typesense `products_v2` (1024-dim) | upsert by product_id |
| 5 ground truth | `python -m make_ground_truth --authorize` | `avtc_metrics.ground_truth` (source='llm') | judge cache (INSERT IGNORE) |
| 6 experiment | `enqueue run_experiment_task <id> <strategy>` | `recommendations` + `compatibility_evaluations` | `EXPERIMENT_ID` keyed |
| 7 evaluate | `python -m run_eval <id>` + honest SQL | P/R/NDCG, precision vs judge GT | pure read, $0 |

## How ground-truth pairs are created (the important part)

`make_ground_truth.py` is the single, automated, reproducible source of truth:

1. **Candidate universe (fixed, config-independent).** For every anchor product
   (a product in an anchor category) the candidates are all products in that
   anchor's *complementary* categories, taken from `category_complements.json`.
   This depends only on (dataset, complement map) - NOT on the retrieval settings
   (`top_k`, `tau_*`, `strategy`). Any experiment config is therefore scored
   against the SAME ground truth.
2. **LLM judge.** Each (anchor, candidate) pair is labelled 1/0 with a rationale
   by `JUDGE_MODEL` (`gpt-5-mini`) reading the REAL product name/type/attributes.
   This catches model-level facts a category graph cannot (e.g. an iPhone has no
   microSD slot, so phone -> memory-card is judged 0 for that model).
3. **Cached + resumable.** Rows are written with `INSERT IGNORE` on
   `(product_i, product_j, source)`; re-running judges only not-yet-labelled pairs.
4. **Budget-guarded.** `--max-cost` (USD) stops the run before the budget is hit;
   `--dry-run` prints universe size and projected cost without spending.

Independence note: the judge GT is **not** the retrieval complement map - it is an
independent per-pair LLM judgement, so precision/recall are meaningful (not
trivially 1.0).

## Engine configuration (viable production config)

In `.env`:

```
COMPAT_MODE=oneshot     # LLM holistic verify (jit rule-mode yields L=0 on this
                        # attribute-poor catalog - no compatibility attributes)
RETRIEVAL_STRATEGY=cat_priors
COMPAT_TOP_K=60
COMPAT_TAU_S=0.15
COMPAT_TAU_L=0.5
JUDGE_MODEL=gpt-5-mini
PRIMARY_MODEL=gpt-5-nano
```

## Engine fixes that made it viable (2026-06-08)

- **strategy propagation**: the worker now sends `strategy` to the engine; it was
  silently ignored before, so every run used the configured strategy.
- **cat_priors no longer gated by `tau_S`**: semantic similarity to the source is
  low for complementary items; the complement map is the relevance signal.
- **strategy-aware verdict**: for cat_priors the verdict rests on logical
  compatibility `L >= tau_L`, not on semantic similarity.
- **`filter_candidates` same-category bug (the big one)**: a phone and its case
  share the broad parent category, so the old "drop any shared category" filter
  discarded real accessories; cat_priors now drops only self.
