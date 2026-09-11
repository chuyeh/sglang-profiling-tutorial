#!/usr/bin/env bash
# In-container AgentX client. Sourced environment comes from docker run / docker exec.
set -euo pipefail
set -x

cd /inferencex
# shellcheck disable=SC1091
source /inferencex/benchmarks/benchmark_lib.sh

check_env_vars \
    MODEL TP CONC EP_SIZE KV_OFFLOADING \
    TOTAL_CPU_DRAM_GB RESULT_DIR DURATION

mkdir -p "$RESULT_DIR" \
    "${HF_DATASETS_CACHE:-$RESULT_DIR/hf_datasets_cache}" \
    "${AIPERF_DATASET_MMAP_CACHE_DIR:-$RESULT_DIR/aiperf_mmap_cache}"

install_agentic_deps

# resolve_trace_source always calls `hf download`. The reconstructed hub cache
# should satisfy huggingface_hub; if offline mode still fails, keep the loader
# flag so aiperf can read traces.jsonl from the snapshot symlink.
set +e
resolve_trace_source
resolve_rc=$?
set -e
if [[ $resolve_rc -ne 0 ]]; then
    echo "WARNING: resolve_trace_source exited $resolve_rc; using WEKA_LOADER_OVERRIDE=$WEKA_LOADER_OVERRIDE" >&2
    TRACE_SOURCE_FLAG="--public-dataset $WEKA_LOADER_OVERRIDE"
fi

# aiperf --tokenizer uses $MODEL. Point it at the local checkpoint while keeping
# the OpenAI wire name on SERVED_MODEL_NAME (set by the launch container env).
export MODEL="${MODEL_PATH:-$MODEL}"

build_replay_cmd "$RESULT_DIR"
REPLAY_CMD+=" --apply-chat-template"
run_agentic_replay_and_write_outputs "$RESULT_DIR"
