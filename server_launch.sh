#!/usr/bin/env bash
set -euo pipefail

MODEL="/models/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81096f3901c91dea97f81324302495781d/"

# python3 -m sglang.launch_server
CUDA_VISIBLE_DEVICES=0,1,2,3 \
sglang serve \
  --model-path "${MODEL}" --tp 4 --ep-size 1 \
  --attention-backend trtllm_mha --moe-runner-backend flashinfer_trtllm \
  --quantization fp8 --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype bfloat16 \
  --enable-symm-mem --trust-remote-code \
  --chunked-prefill-size 32768 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --mem-fraction-static 0.9 \
  --host 0.0.0.0 --port 9001 --disable-radix-cache \
  --max-running-requests 512 \
  --page-size 16 \
  2>&1 | tee "tmp_qwen35_b200_accuracy_$(date +%Y%m%d_%H%M%S).log"
