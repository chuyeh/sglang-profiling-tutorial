#!/usr/bin/env bash
set -euo pipefail

MODEL="/raid/models/Qwen3.5-397B-A17B-FP8/"
ISL=8192
OSL=1024
CONTEXT_LENGTH=$((ISL + OSL + 20))
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
DISABLE_FUSED_AR_QUANT="${SGLANG_DISABLE_FUSED_AR_QUANT:-0}"

if [[ "${DISABLE_FUSED_AR_QUANT}" == "1" ]]; then
  export SGLANG_DISABLE_FUSED_AR_QUANT=1
else
  unset SGLANG_DISABLE_FUSED_AR_QUANT
fi

# python3 -m sglang.launch_server
ROCR_VISIBLE_DEVICES="${VISIBLE_GPUS}" \
SGLANG_USE_AITER_UNIFIED_ATTN=1 \
SGLANG_USE_AITER=1 \
sglang serve \
  --host 0.0.0.0 \
  --port 9001 \
  --model-path "${MODEL}" \
  --tp "${TP_SIZE}" \
  --ep-size 1 \
  --attention-backend aiter \
  --trust-remote-code \
  --chunked-prefill-size 8192 \
  --model-loader-extra-config '{"enable_multithread_load": true}' \
  --watchdog-timeout 1200 \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --tokenizer-worker-num 6 \
  --disable-radix-cache \
  --enable-aiter-allreduce-fusion \
  --page-size 16 \
  --kv-cache-dtype fp8_e4m3 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --context-length $CONTEXT_LENGTH

  # 2>&1 | tee "tmp_qwen35_mi35x_accuracy_$(date +%Y%m%d_%H%M%S).log"