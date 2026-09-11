#!/usr/bin/env bash
# Run one Qwen3.5 AgentX point and capture a short SGLang torch profile.
#
# Usage:
#   run_profile_point.sh <label> <TP> <CONC> <GPUS> <none|dram>
#
# Defaults specific to profiling:
#   DURATION=1500, PROFILE_DELAY_S (or WARM)=840, PROFILE_WINDOW_S
#   (or WINDOW)=12. The delay is measured from the start of AIPerf's measured
#   phase, not from server health.
set -Eeuo pipefail

if [[ "$#" -ne 5 ]]; then
    echo "Usage: $0 <label> <TP> <CONC> <GPUS> <none|dram>" >&2
    exit 2
fi

label="$1"
tp="$2"
conc="$3"
gpus="${4//[[:space:]]/}"
kv="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

[[ "$label" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "ERROR: label must contain only letters, numbers, '.', '_' and '-'" >&2
    exit 2
}
[[ "$tp" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: TP must be a positive integer" >&2
    exit 2
}
[[ "$conc" =~ ^[1-9][0-9]*$ ]] || {
    echo "ERROR: CONC must be a positive integer" >&2
    exit 2
}

IFS=',' read -ra gpu_ids <<< "$gpus"
if [[ "${#gpu_ids[@]}" -ne "$tp" ]]; then
    echo "ERROR: GPUS contains ${#gpu_ids[@]} devices, but TP=$tp (these must match)" >&2
    exit 2
fi
declare -A seen_gpus=()
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid GPU id: $gpu" >&2
        exit 2
    }
    [[ -z "${seen_gpus[$gpu]:-}" ]] || {
        echo "ERROR: duplicate GPU id: $gpu" >&2
        exit 2
    }
    seen_gpus["$gpu"]=1
done

case "$kv" in
    none)
        export KV_OFFLOADING=none
        export KV_OFFLOAD_BACKEND=
        export TOTAL_CPU_DRAM_GB=0
        ;;
    dram)
        dram_gb="${TOTAL_CPU_DRAM_GB:-${DRAM_GB:-}}"
        [[ "$dram_gb" =~ ^[1-9][0-9]*$ ]] || {
            echo "ERROR: set TOTAL_CPU_DRAM_GB (or DRAM_GB) to the allowed host-DRAM budget for a dram run" >&2
            exit 2
        }
        export KV_OFFLOADING=dram
        export KV_OFFLOAD_BACKEND=hicache
        export TOTAL_CPU_DRAM_GB="$dram_gb"
        ;;
    *)
        echo "ERROR: invalid KV mode '$kv'; use none or dram" >&2
        exit 2
        ;;
esac

run_id="${label}_tp${tp}_c${conc}_${kv}"
campaign_root="${AGENTX_RUNS_ROOT:-$REPO_ROOT/campaigns/agentx-torch-profiles-$(date +%Y%m%d)}"
export RESULT_DIR="${RESULT_DIR:-$campaign_root/$run_id}"

if [[ -d "$RESULT_DIR" ]]; then
    shopt -s nullglob dotglob
    existing=("$RESULT_DIR"/*)
    shopt -u nullglob dotglob
    if (( ${#existing[@]} > 0 )); then
        echo "ERROR: RESULT_DIR is not empty: $RESULT_DIR" >&2
        echo "Use a new label or set RESULT_DIR to a new directory." >&2
        exit 2
    fi
fi
mkdir -p "$RESULT_DIR"

if [[ -z "${PORT:-}" ]]; then
    read -r checksum _ <<< "$(printf '%s' "$run_id" | cksum)"
    export PORT=$((18000 + checksum % 20000))
fi

export TP="$tp"
export CONC="$conc"
export GPUS="$gpus"
export CONTAINER_NAME="${CONTAINER_NAME:-agentx-profile-${run_id}}"
export DURATION="${DURATION:-1500}"
export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-10}"
export AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-false}"
export SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-3600}"
export ENABLE_TORCH_PROFILER=1
export PROFILE_DELAY_S="${PROFILE_DELAY_S:-${WARM:-840}}"
export PROFILE_WINDOW_S="${PROFILE_WINDOW_S:-${WINDOW:-12}}"
export PROFILE_MIN_RUNNING_REQUESTS="${PROFILE_MIN_RUNNING_REQUESTS:-1}"

# Physical IDs are filtered by ROCR_VISIBLE_DEVICES. Let env.sh remap HIP IDs
# to 0..TP-1 unless the caller deliberately supplies a mapping.
if [[ -z "${PRESERVE_HIP_VISIBLE_DEVICES:-}" ]]; then
    unset HIP_VISIBLE_DEVICES
fi

{
    echo "RUN_ID=$run_id"
    echo "TP=$TP"
    echo "CONC=$CONC"
    echo "GPUS=$GPUS"
    echo "KV_OFFLOADING=$KV_OFFLOADING"
    echo "KV_OFFLOAD_BACKEND=$KV_OFFLOAD_BACKEND"
    echo "TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB"
    echo "PORT=$PORT"
    echo "DURATION=$DURATION"
    echo "PROFILE_DELAY_S=$PROFILE_DELAY_S"
    echo "PROFILE_WINDOW_S=$PROFILE_WINDOW_S"
    echo "PROFILE_MIN_RUNNING_REQUESTS=$PROFILE_MIN_RUNNING_REQUESTS"
    echo "RESULT_DIR=$RESULT_DIR"
} > "$RESULT_DIR/profile_point_env.txt"

echo "Launching $run_id on physical GPUs $GPUS (port=$PORT)"
"$SCRIPT_DIR/qwen3.5_fp4_mi355x_launch_server.sh"

set +e
"$SCRIPT_DIR/qwen3.5_fp4_mi355x_run_client.sh"
rc=$?
set -e

if [[ "${STOP_CONTAINER_AFTER:-0}" == "1" ]]; then
    echo "Stopping $CONTAINER_NAME"
    docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
else
    echo "Container left running: $CONTAINER_NAME"
    echo "Stop later with: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"
fi

if [[ "$rc" -eq 0 ]]; then
    echo "Profile point complete: $RESULT_DIR"
else
    echo "ERROR: profile point failed (rc=$rc): $RESULT_DIR" >&2
fi
exit "$rc"
