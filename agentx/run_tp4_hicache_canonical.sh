#!/usr/bin/env bash
# Canonical InferenceX-style TP4 HiCache sweep for Qwen3.5 MXFP4 on MI355X.
# One server launch per conc (CUDA-graph BS = min(2*CONC, 128)).
# Do not source env.sh here: that file's smoke defaults would override these
# assignments if we used ${VAR:-...} after sourcing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONCS=(40 48 56 64)
CAMPAIGN_DIR="$REPO_ROOT/campaigns/agentx-canonical-hicache-tp4-$(date +%Y%m%d)"
STATUS_LOG="$CAMPAIGN_DIR/sweep_status.log"
mkdir -p "$CAMPAIGN_DIR"

# Unset HIP so env.sh derives 0,1,2,3 after ROCR remaps GPUS 4-7.
unset HIP_VISIBLE_DEVICES

export IMAGE="rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260906"
export TP=4
export EP_SIZE=1
export KV_OFFLOADING=dram
export KV_OFFLOAD_BACKEND=hicache
export TOTAL_CPU_DRAM_GB=1199
export DURATION=3600
export AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_UNSAFE_OVERRIDE=false
export ENABLE_AGENTX_POWER=1
export GPUS=4,5,6,7
export PORT=18988
export SERVER_READY_TIMEOUT_S=3600
export FRAMEWORK=sglang
export PRECISION=fp4
export SPEC_DECODING=mtp
export RUNNER_TYPE=cluster:mi355x-amds

log() {
    printf '%s %s\n' "$(date -Iseconds)" "$*" | tee -a "$STATUS_LOG"
}

stop_container() {
    local name="$1"
    if docker inspect "$name" >/dev/null 2>&1; then
        log "Stopping $name"
        docker stop -t 60 "$name" >/dev/null 2>&1 || true
        docker rm -f "$name" >/dev/null 2>&1 || true
    fi
}

{
    echo "IMAGE=$IMAGE"
    echo "GPUS=$GPUS"
    echo "CONCS=${CONCS[*]}"
    echo "DURATION=$DURATION"
    echo "AIPERF_WARMUP_REQUESTS_PER_LANE=$AIPERF_WARMUP_REQUESTS_PER_LANE"
    echo "AIPERF_UNSAFE_OVERRIDE=$AIPERF_UNSAFE_OVERRIDE"
    echo "ENABLE_AGENTX_POWER=$ENABLE_AGENTX_POWER"
    echo "CAMPAIGN_DIR=$CAMPAIGN_DIR"
} | tee "$CAMPAIGN_DIR/sweep_env.txt"

overall_rc=0
for conc in "${CONCS[@]}"; do
    export CONC="$conc"
    export CONTAINER_NAME="agentx-qwen35-hicache-tp4-c${conc}"
    export RESULT_DIR="$CAMPAIGN_DIR/c${conc}"
    mkdir -p "$RESULT_DIR"
    log "===== START conc=$conc container=$CONTAINER_NAME image=$IMAGE duration=$DURATION warmup=$AIPERF_WARMUP_REQUESTS_PER_LANE ====="

    stop_container "$CONTAINER_NAME"
    set +e
    "$SCRIPT_DIR/qwen3.5_fp4_mi355x_launch_server.sh"
    launch_rc=$?
    set -e
    if [[ "$launch_rc" -ne 0 ]]; then
        log "FAIL launch conc=$conc rc=$launch_rc"
        stop_container "$CONTAINER_NAME"
        overall_rc=1
        continue
    fi

    set +e
    "$SCRIPT_DIR/qwen3.5_fp4_mi355x_run_client.sh"
    client_rc=$?
    set -e
    if [[ "$client_rc" -ne 0 ]]; then
        log "FAIL client conc=$conc rc=$client_rc"
        overall_rc=1
    else
        log "OK conc=$conc artifacts=$RESULT_DIR"
    fi
    stop_container "$CONTAINER_NAME"
done

log "===== SWEEP DONE overall_rc=$overall_rc ====="
exit "$overall_rc"
