#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="9001"
MODEL="/models/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d/"


DATASET="random"

input_tokens=8192
output_tokens=1024

random_range_ratio=1.0

# Where to save outputs
CSV_OUT="bench_results.csv"
PLOT_OUT="throughput_vs_median_e2e_latency.png"
# =================================================================

concurrencies=(4)

# Start fresh CSV
echo "concurrency,median_e2e_ms,total_token_throughput_tok_s" > "$CSV_OUT"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

for c in "${concurrencies[@]}"; do
  max_concurrency="$c"
  # num_prompts=$((max_concurrency * 8))
  num_prompts=$((max_concurrency * 2))

  echo "=== Running benchmark: concurrency=${c}, num_prompts=${num_prompts} ==="
  tmp_log="server-mode_benchmark_results_max_concurrency${max_concurrency}_${RUN_TIMESTAMP}.log"
  # Run the benchmark and capture output
  # SGLANG_TORCH_PROFILER_DIR=/sgl-workspace/sglang/python/baseline_conc1_offline

  # curl -X POST http://localhost:9000/start_profile \
  #   -H "Content-Type: application/json" \
  #   -d '{
  #     "output_dir": "/tmp/profiles/rpd",
  #     "activities": ["RPD"]
  #   }'

  python3 -m sglang.bench_serving \
      --host "${HOST}" \
      --port "${PORT}" \
      --model "${MODEL}" \
      --dataset-name "${DATASET}" \
      --random-input-len "${input_tokens}" \
      --random-output-len "${output_tokens}" \
      --random-range-ratio "${random_range_ratio}" \
      --max-concurrency "${max_concurrency}" \
      --profile \
      --profile-output-dir /workspace/profiling/output \
      --num-prompt "${num_prompts}" 2>&1 | tee "${tmp_log}"
  # python -m sglang.test.send_one --port 9000
  # curl http://localhost:9000/stop_profile -H "Content-Type: application/json"

  # export CSV_FILE="trace.csv"
  # sqlite3 ./trace.rpd ".mode csv" ".header on" ".output $CSV_FILE" "select * from top;" ".output stdout"
  # sleep 10
done
