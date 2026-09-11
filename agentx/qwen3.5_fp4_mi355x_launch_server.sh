#!/usr/bin/env bash
set -euo pipefail

# Launch the Qwen3.5 MXFP4 MI355X AgentX SGLang server in Docker and leave it up.
# Mirrors InferenceX benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh
# server flags without the client or the EXIT-trap teardown.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

agentx_validate_gpu_selection
agentx_require_paths
agentx_prepare_hf_cache
mkdir -p "$RESULT_DIR"
if agentx_bool_enabled "$ENABLE_TORCH_PROFILER"; then
    if [[ "$TORCH_PROFILE_SUBDIR" == /* || "$TORCH_PROFILE_SUBDIR" == *".."* ]]; then
        echo "ERROR: TORCH_PROFILE_SUBDIR must be a safe relative path under RESULT_DIR" >&2
        exit 1
    fi
    mkdir -p "$RESULT_DIR/$TORCH_PROFILE_SUBDIR"
    echo "Torch profiler armed; traces will be written to $RESULT_DIR/$TORCH_PROFILE_SUBDIR"
fi

MAX_RUNNING_REQUESTS=$((2 * CONC))
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
if [ "$CUDA_GRAPH_MAX_BS" -gt 128 ]; then
    CUDA_GRAPH_MAX_BS=128
fi

CACHE_ARGS=()
if [[ "$KV_OFFLOADING" == "dram" ]]; then
    if [[ "${KV_OFFLOAD_BACKEND:-}" != "hicache" ]]; then
        echo "ERROR: KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=hicache" >&2
        exit 1
    fi
    HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
    HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
    HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
    HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
    echo "HiCache CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
    CACHE_ARGS=(
        --enable-hierarchical-cache
        --hicache-ratio "$HICACHE_RATIO"
        --hicache-write-policy "$HICACHE_WRITE_POLICY"
        --hicache-io-backend "$HICACHE_IO_BACKEND"
        --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    )
fi

TOKENIZER_ARGS=()
if [ "$TP" -ge 4 ]; then
    TOKENIZER_ARGS=(--tokenizer-worker-num 6)
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    --tp "$TP"
    --dp 1
    --ep-size "$EP_SIZE"
    --attention-backend aiter
    --mem-fraction-static 0.80
    --model-loader-extra-config '{"enable_multithread_load": true}'
    --watchdog-timeout 1200
    --page-size 16
    --kv-cache-dtype fp8_e4m3
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --max-prefill-tokens 16384
    --chunked-prefill-size 16384
    --scheduler-recv-interval "$SCHEDULER_RECV_INTERVAL"
    --stream-interval 50
    "${TOKENIZER_ARGS[@]}"
    --tokenizer-path "$MODEL_PATH"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
    --enable-metrics
    --enable-cache-report
    "${CACHE_ARGS[@]}"
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

DOCKER_ARGS=(
    docker run -d
    --name "$CONTAINER_NAME"
    --pid=host
    --network host
    --device=/dev/kfd
    --device=/dev/dri
    --ipc=host
    --shm-size=32g
    --cap-add SYS_PTRACE
    --security-opt seccomp=unconfined
    -v "$DATA_ROOT":"$DATA_ROOT":ro
    -v "$INFERENCEX_ROOT":/inferencex:ro
    -v "$AIPERF_HOST":/opt/aiperf:ro
    -v "$RESULT_DIR":/results
    -v "$SCRIPT_DIR":/opt/agentx:ro
    -v "$HF_HOME":/hf_home
    -w /inferencex
)

# Device group IDs vary across MI355X hosts; derive them instead of assuming
# the local server's video/render GIDs.
declare -A DEVICE_GROUP_GIDS=()
for device_path in /dev/kfd /dev/dri/renderD* /dev/dri/card*; do
    [[ -e "$device_path" ]] || continue
    device_gid="$(stat -c '%g' "$device_path")"
    DEVICE_GROUP_GIDS["$device_gid"]=1
done
for device_gid in "${!DEVICE_GROUP_GIDS[@]}"; do
    DOCKER_ARGS+=(--group-add "$device_gid")
done
unset DEVICE_GROUP_GIDS device_path device_gid

if [[ -n "$SGLANG_ROOT" ]]; then
    DOCKER_ARGS+=(-v "$SGLANG_ROOT":/experiment:ro)
fi

while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    DOCKER_ARGS+=(-e "$line")
done < <(agentx_container_env)

DOCKER_ARGS+=("$IMAGE" "${SGLANG_CMD[@]}")

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    echo "Removing existing container $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting $CONTAINER_NAME on GPUs=$GPUS port=$PORT"
CONTAINER_ID="$("${DOCKER_ARGS[@]}")"
printf '%s\n' "$CONTAINER_ID" > "$RESULT_DIR/container_id.txt"
printf '%s\n' "$CONTAINER_NAME" > "$RESULT_DIR/container_name.txt"

dump_logs() {
    docker logs "$CONTAINER_NAME" > "$RESULT_DIR/server.log" 2>&1 || true
}

deadline=$((SECONDS + SERVER_READY_TIMEOUT_S))
ready=0
while (( SECONDS < deadline )); do
    if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; then
        dump_logs
        echo "ERROR: $CONTAINER_NAME exited before becoming ready. See $RESULT_DIR/server.log" >&2
        exit 1
    fi
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        ready=1
        break
    fi
    dump_logs
    sleep 10
done
dump_logs

if [[ "$ready" -ne 1 ]]; then
    echo "ERROR: server not ready after ${SERVER_READY_TIMEOUT_S}s. See $RESULT_DIR/server.log" >&2
    exit 1
fi

echo "Server ready on http://127.0.0.1:${PORT}"
curl -fsS "http://127.0.0.1:${PORT}/v1/models" | tee "$RESULT_DIR/models.json"
printf '\n'

smoke_payload='{"model":"'"$SERVED_MODEL_NAME"'","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":8,"temperature":0,"stream":true}'
if curl -fsS "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$smoke_payload" \
    | tee "$RESULT_DIR/smoke_chat.ndjson" >/dev/null; then
    echo "Smoke streaming chat request succeeded. Artifacts in $RESULT_DIR"
else
    echo "WARNING: smoke chat request failed; server is still running. See $RESULT_DIR/server.log" >&2
fi

echo "Container $CONTAINER_NAME left running. Use agentx/qwen3.5_fp4_mi355x_run_client.sh next."
echo "Stop later with: docker stop $CONTAINER_NAME"
