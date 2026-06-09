#!/usr/bin/env bash
# Reproduce the full AVTC cross-sell experiment end to end ("2 clicks").
#
#   click 1:  ./reproduce.sh data     # dataset -> catalog -> normalize -> embed -> ground truth
#   click 2:  ./reproduce.sh run      # experiment -> evaluation (prints P/R/NDCG)
#   or:       ./reproduce.sh all      # everything in sequence
#
# Everything is deterministic and idempotent:
#   - dataset build uses SEED=42
#   - normalize/embed skip already-processed products (source_hash / upsert)
#   - ground truth is judge-cached (INSERT IGNORE; reruns judge only new pairs)
#   - experiments are keyed by EXPERIMENT_ID so reruns overwrite cleanly
#
# Ground truth is created automatically by an LLM judge (JUDGE_MODEL), never by
# hand. See make_ground_truth.py.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root (E:/AVTC)

EXPERIMENT_ID="${EXPERIMENT_ID:-cat_priors_v1}"
STRATEGY="${STRATEGY:-cat_priors}"
JUDGE_MAX_COST="${JUDGE_MAX_COST:-8}"

db()   { docker exec avtc_db mysql -uuser -psecret "$@"; }
eng()  { docker exec avtc_engine python -m "$@"; }

stage_data() {
  echo "== 1/5 build curated dataset (deterministic) =="
  python benchmark/build.py
  cp benchmark/products.json   project/dataset/json/products.json
  cp benchmark/categories.json project/dataset/json/categories.json
  cp benchmark/attributes.json project/dataset/json/attributes.json

  echo "== 2/5 start infra + import catalog =="
  docker compose up -d db typesense redis engine worker api
  sleep 8
  eng catalog_importer

  echo "== 3/5 normalize (idempotent) =="
  eng normalize

  echo "== 4/5 (re)build embeddings in Typesense =="
  eng index_products init --collection products_v2 --recreate
  eng index_products run  --collection products_v2

  echo "== 5/5 ground truth via LLM judge (auto, reproducible) =="
  eng make_ground_truth --authorize --batch 15 --concurrency 8 --max-cost "$JUDGE_MAX_COST"
}

stage_run() {
  echo "== experiment: $EXPERIMENT_ID strategy=$STRATEGY =="
  db avtc_metrics -e "DELETE FROM recommendations WHERE experiment_id='$EXPERIMENT_ID';
                      DELETE FROM compatibility_evaluations WHERE experiment_id='$EXPERIMENT_ID';"
  docker exec avtc_worker python -m enqueue run_experiment_task "$EXPERIMENT_ID" "$STRATEGY"

  echo "waiting for the worker to drain..."
  while [ "$(docker logs avtc_worker --since 12s 2>&1 | grep -c 'process_product_task ●')" != "0" ]; do
    sleep 8
  done

  echo "== built-in evaluation (P/R/NDCG@K) =="
  eng run_eval "$EXPERIMENT_ID" || true

  echo "== honest precision vs judge ground truth (source='llm') =="
  db avtc_metrics -N -e "
    SELECT
      (SELECT COUNT(*) FROM recommendations WHERE experiment_id='$EXPERIMENT_ID' AND verdict=1) AS predicted,
      (SELECT COUNT(*) FROM recommendations r JOIN ground_truth g
         ON g.product_i=r.product_id AND g.product_j=r.recommended_id AND g.source='llm' AND g.label=1
       WHERE r.experiment_id='$EXPERIMENT_ID' AND r.verdict=1) AS true_pos,
      (SELECT COUNT(*) FROM ground_truth WHERE source='llm' AND label=1) AS gt_positive;" \
    | awk '{printf "predicted=%s true_pos=%s gt_pos=%s precision=%.3f recall=%.3f\n",$1,$2,$3,($1?$2/$1:0),($3?$2/$3:0)}'
}

case "${1:-all}" in
  data) stage_data ;;
  run)  stage_run ;;
  all)  stage_data; stage_run ;;
  *) echo "usage: $0 {data|run|all}"; exit 1 ;;
esac
echo "done."
