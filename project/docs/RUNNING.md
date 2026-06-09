# Running the full recommendation pipeline (local, $0)

Everything runs locally via Ollama (verification = `gemma4:e4b-it-q8_0`, embedding =
`dengcao/Qwen3-Embedding-0.6B:Q8_0`) + Typesense + MySQL. No cloud. The run is heavy on
the GPU, so launch it when you are **away from the PC** (not gaming).

The run is **resumable**: stop it any time, start it again with `--resume` and it skips
products that already have recommendations and continues.

## 0. Prerequisites (once)

- Ollama running with both models pulled:
  - `ollama pull hf.co/lmstudio-community/gemma-4-e4b-it-GGUF:Q8_0` (or `gemma4:e4b-it-q8_0`)
  - `ollama pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0`
  - keep only one model in VRAM at a time: `setx OLLAMA_MAX_LOADED_MODELS 1`
- Containers up:
  ```
  cd backend && docker compose up -d
  ```

## 1. Start the full run (resumable, detached, survives terminal close)

```
docker exec -d avtc_backend sh -c "python -m app.compatibility --resume > /tmp/compat.log 2>&1"
```

First clean run (wipe old recommendations) - skip `--resume` and truncate first:
```
docker exec avtc_db mysql -uroot -proot -e "TRUNCATE avtc_metrics.recommendations; TRUNCATE avtc_metrics.compatibility_evaluations;"
docker exec -d avtc_backend sh -c "python -m app.compatibility > /tmp/compat.log 2>&1"
```

Convenience script (same thing):
```
bash backend/run_compat.sh          # resumable full run
bash backend/run_compat.sh fresh    # wipe + full run from scratch
```

## 2. Monitor

```
docker exec avtc_backend sh -c "grep -vE 'HTTP Request' /tmp/compat.log | tail -15"
docker exec avtc_db mysql -uroot -proot -N -e "SELECT COUNT(*), SUM(verdict), COUNT(DISTINCT product_id) FROM avtc_metrics.recommendations;"
```

Live coverage on the storefront: http://avtc.local/#/metrics

## 3. Stop (e.g. you want to game)

```
docker restart avtc_backend
```
This kills the detached run. Re-launch later with `--resume` to continue.
To free GPU immediately, also stop/exit Ollama or let `OLLAMA_KEEP_ALIVE` expire.

## 4. Throughput

gemma4-e4b Q8 on a 12 GB GPU: ~8 s per 10-candidate batch (warm), concurrency 4.
Full 14 767 products ≈ overnight. Partial runs are fine - coverage grows incrementally
and the storefront shows recommendations for whatever is done.

## 5. After the run - evaluation metrics (TODO, not built yet)

The eval engine (`python -m app.eval`, Stage 4 of the plan) will compute Precision/Recall/
NDCG@K and the alpha curve from attribute/category ground truth and write them to
`avtc_metrics.quality_snapshots` / `alpha_experiments`, filling the empty charts on the
`/metrics` page. Until then `/metrics` shows real catalog / coverage / context_code
distribution, with P/R/NDCG/alpha at zero.
