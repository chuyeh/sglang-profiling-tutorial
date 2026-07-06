#!/usr/bin/env bash
set -euo pipefail

MODEL="/raid/models/Qwen3.5-397B-A17B-FP8/"

# python3 -m sglang.launch_server
HIP_VISIBLE_DEVICES=0,1,2,3 \
SGLANG_USE_AITER_UNIFIED_ATTN=1 SGLANG_USE_AITER=1 \
sglang serve \
  --model-path "${MODEL}" --tp 4 \
  --attention-backend aiter --trust-remote-code \
  --chunked-prefill-size 32768 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 --mem-fraction-static 0.9 \
  --host 0.0.0.0 --port 9001 --disable-radix-cache \
  --enable-aiter-allreduce-fusion --max-running-requests 512 \
  --page-size 16
  2>&1 | tee "tmp_qwen35_mi35x_accuracy_$(date +%Y%m%d_%H%M%S).log"
 
