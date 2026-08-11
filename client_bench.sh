#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="9001"
MODEL="/raid/models/Qwen3.5-397B-A17B-FP8/"

DATASET="random"

input_tokens=8192
output_tokens=1024

random_range_ratio=0.8

# Set ENABLE_PROFILE=1 only when collecting a trace. Profiling changes
# end-to-end performance, so leave it off for the PR #24651 speed comparison.
ENABLE_PROFILE="${ENABLE_PROFILE:-1}"

# Where to save outputs
# CSV_OUT="bench_results.csv"
PLOT_OUT="throughput_vs_median_e2e_latency.png"
# =================================================================

concurrencies=(4)

# Start fresh CSV
# echo "concurrency,median_e2e_ms,total_token_throughput_tok_s" > "$CSV_OUT"
RUN_TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"

for c in "${concurrencies[@]}"; do
  max_concurrency="$c"
  # num_prompts=$((max_concurrency * 8))
  num_prompts=$((max_concurrency * 10))

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

  profile_args=()
  if [[ "${ENABLE_PROFILE}" == "1" ]]; then
    profile_args=(
      --profile
      --profile-output-dir /workspace/profiling/output/mi355
    )
  fi

  python3 -m sglang.bench_serving \
    --host "${HOST}" \
    --port "${PORT}" \
    --model "${MODEL}" \
    --dataset-name "${DATASET}" \
    --random-input "${input_tokens}" \
    --random-output "${output_tokens}" \
    --random-range-ratio "${random_range_ratio}" \
    --max-concurrency "${max_concurrency}" \
    --num-prompts "${num_prompts}" \
    "${profile_args[@]}" 2>&1 | tee "${tmp_log}"
  # python -m sglang.test.send_one --port 9000
  # curl http://localhost:9000/stop_profile -H "Content-Type: application/json"

  # export CSV_FILE="trace.csv"
  # sqlite3 ./trace.rpd ".mode csv" ".header on" ".output $CSV_FILE" "select * from top;" ".output stdout"
  # sleep 10
done
