#!/usr/bin/env bash
# One-command, deadline-aware Qwen3.5 MXFP4 AgentX sweep for an 8xMI355X node.
#
# Default: a 16-point, one-reservation subset. Use --mode full for all 24
# official points; progress is checkpointed and the same command resumes it.
set -Eeuo pipefail

# -----------------------------------------------------------------------------
# Runtime paths (derived automatically; do not edit)
# -----------------------------------------------------------------------------
readonly SCRIPT_START_EPOCH="$(date +%s)"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
readonly MACHINE_CONFIG="$SCRIPT_DIR/machine.conf"
readonly CURRENT_USER="${USER:-$(id -un)}"

# -----------------------------------------------------------------------------
# Reproducibility lock (maintainer-owned)
#
# These pins identify one reviewed experiment definition. Normal users should
# use CLI options or machine.conf for local paths and should not edit this
# block. When intentionally upgrading a dependency:
#   1. review the upstream changes;
#   2. update its revision and checksum(s) together;
#   3. validate with --plan and --preflight; and
#   4. use a new campaign directory so old and new results cannot mix.
#
# Sources:
#   InferenceX: https://github.com/SemiAnalysisAI/InferenceX
#   Model:      https://huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4
#   Traces:     https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k
# Last reviewed: 2026-09-12
# -----------------------------------------------------------------------------
readonly PINNED_INFERENCEX_REVISION="9acf24caaf31d9ecc5067742d6d1415967b308ed"
readonly PINNED_AIPERF_REVISION="754356e9a39acc6cc6afb242d123bb57c3fb6f75"
readonly PINNED_IMAGE="rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260911"
readonly PINNED_MODEL_ID="amd/Qwen3.5-397B-A17B-MXFP4"
readonly PINNED_MODEL_REVISION="edf0958bc3734dda98a9d191cc7a0a83c4f42821"
# Local model directories may lack Git metadata, so verify their two small
# identity files without hashing hundreds of gigabytes of weight shards.
readonly PINNED_MODEL_CONFIG_SHA256="f3f18d5b47ac8d980ecb8dfc182a4a545de65913bfd007c5facd0f604c23e0b0"
readonly PINNED_MODEL_INDEX_SHA256="c1e4f3d24049deb621cfbb2740131f5361d3ec3e9ee0d8ca906ca0529e02cd00"
readonly PINNED_TRACE_REVISION="8fecd2fc56694469f758f0afbbb6335ad3043740"
# The trace file is small enough to verify completely.
readonly PINNED_TRACE_SHA256="e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef"

# -----------------------------------------------------------------------------
# Canonical benchmark settings (must match the reviewed InferenceX recipe)
# -----------------------------------------------------------------------------
readonly CANONICAL_DURATION_S=3600
readonly CANONICAL_WARMUP_REQUESTS_PER_LANE=10
readonly CANONICAL_SYNTHETIC_ACCEPTANCE_LENGTH=3.39
readonly CANONICAL_FAILED_REQUEST_THRESHOLD=0.10
readonly CANONICAL_WARMUP_GRACE_S=1800
readonly CANONICAL_TRACE_IDLE_GAP_CAP_S=300
readonly CANONICAL_SERVER_READY_TIMEOUT_S=3600
readonly CANONICAL_SCHEDULER_RECV_INTERVAL=30
readonly CANONICAL_HICACHE_RATIO=1.5
readonly CANONICAL_HICACHE_WRITE_POLICY="write_through"
readonly CANONICAL_HICACHE_IO_BACKEND="direct"
readonly CANONICAL_HICACHE_MEM_LAYOUT="page_first_direct"
readonly TP2_HICACHE_DRAM_GB=599
readonly TP4_HICACHE_DRAM_GB=1199

# -----------------------------------------------------------------------------
# Local scheduling and safety policy (not model/recipe identity)
# -----------------------------------------------------------------------------
readonly SECONDS_PER_HOUR=3600
readonly DEFAULT_SWEEP_MODE="24h"
readonly DEFAULT_TIME_LIMIT_HOURS=23
readonly DEFAULT_PORT=18988
readonly POINT_START_BUDGET_S=7800
readonly CLEANUP_RESERVE_S=600
readonly ESTIMATED_POINT_MINUTES=78
readonly DEPENDENCY_BOOTSTRAP_TIMEOUT_S=900
readonly PREFLIGHT_COMMAND_TIMEOUT_S=60
readonly IMAGE_PULL_TIMEOUT_S=3600
readonly TIMEOUT_KILL_GRACE_S=30
readonly POINT_KILL_GRACE_S=120
readonly CONTAINER_STOP_TIMEOUT_S=60
readonly MIN_NODE_DRAM_MIB=2861022
readonly MIN_AVAILABLE_NODE_DRAM_KB=1500000000
readonly MIN_CAMPAIGN_FREE_GIB=30
readonly MAX_IDLE_VRAM_PERCENT=10

if [[ -f "$MACHINE_CONFIG" ]]; then
    # shellcheck source=/dev/null
    source "$MACHINE_CONFIG"
fi

MODE="${SWEEP_MODE:-$DEFAULT_SWEEP_MODE}"
TIME_LIMIT_HOURS="${SWEEP_TIME_LIMIT_HOURS:-$DEFAULT_TIME_LIMIT_HOURS}"
PLAN_ONLY=0
PREFLIGHT_ONLY=0
PULL_IMAGE=1

usage() {
    cat <<'EOF'
Usage: run_official_qwen35_agentx_sweep.sh [options]

Options:
  --mode 24h|full          16-point one-booking set (default), or all 24 points
  --hours N                Hard wall-clock budget including setup (default: 23)
  --plan                   Print resolved paths and matrix; run nothing
  --preflight              Validate the machine and campaign; run no points
  --no-pull                Require the official image to exist locally
  --data-root PATH         Override automatic /data2-style discovery
  --model-path PATH        Override automatic model discovery
  --trace-dir PATH         Override automatic AgentX trace discovery
  --inferencex-root PATH   Use an exact pinned InferenceX checkout
  --aiperf-root PATH       Use an exact pinned AIPerf checkout
  --campaign-dir PATH      Override the persistent checkpoint/results directory
  -h, --help               Show this help

The same command is safe to rerun: completed points are validated and skipped.
Machine-specific path overrides can also be saved once in agentx/machine.conf.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            [[ $# -ge 2 ]] || die "--mode needs a value"
            MODE="$2"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --hours)
            [[ $# -ge 2 ]] || die "--hours needs a value"
            TIME_LIMIT_HOURS="$2"
            shift 2
            ;;
        --hours=*)
            TIME_LIMIT_HOURS="${1#*=}"
            shift
            ;;
        --plan)
            PLAN_ONLY=1
            shift
            ;;
        --preflight)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --no-pull)
            PULL_IMAGE=0
            shift
            ;;
        --data-root)
            [[ $# -ge 2 ]] || die "--data-root needs a value"
            SWEEP_DATA_ROOT="$2"
            shift 2
            ;;
        --model-path)
            [[ $# -ge 2 ]] || die "--model-path needs a value"
            SWEEP_MODEL_PATH="$2"
            shift 2
            ;;
        --trace-dir)
            [[ $# -ge 2 ]] || die "--trace-dir needs a value"
            SWEEP_TRACE_DIR="$2"
            shift 2
            ;;
        --inferencex-root)
            [[ $# -ge 2 ]] || die "--inferencex-root needs a value"
            SWEEP_INFERENCEX_ROOT="$2"
            shift 2
            ;;
        --aiperf-root)
            [[ $# -ge 2 ]] || die "--aiperf-root needs a value"
            SWEEP_AIPERF_ROOT="$2"
            shift 2
            ;;
        --campaign-dir)
            [[ $# -ge 2 ]] || die "--campaign-dir needs a value"
            SWEEP_CAMPAIGN_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

[[ "$MODE" == "24h" || "$MODE" == "full" ]] ||
    die "--mode must be 24h or full"
[[ "$TIME_LIMIT_HOURS" =~ ^[1-9][0-9]*$ ]] ||
    die "--hours must be a positive integer"

# -----------------------------------------------------------------------------
# Point sets
#
# FULL_MATRIX mirrors the reviewed InferenceX matrix. ONE_BOOKING_POINT_IDS is
# the 16-point subset selected by --mode 24h.
# -----------------------------------------------------------------------------
declare -ar FULL_MATRIX=(
    "tp2-c1-hbm|2|1|none"
    "tp2-c4-hbm|2|4|none"
    "tp2-c8-hbm|2|8|none"
    "tp2-c12-hbm|2|12|none"
    "tp2-c16-hbm|2|16|none"
    "tp2-c20-hbm|2|20|none"
    "tp2-c20-hicache|2|20|dram"
    "tp2-c24-hicache|2|24|dram"
    "tp2-c28-hicache|2|28|dram"
    "tp2-c32-hicache|2|32|dram"
    "tp4-c1-hbm|4|1|none"
    "tp4-c4-hbm|4|4|none"
    "tp4-c8-hbm|4|8|none"
    "tp4-c12-hbm|4|12|none"
    "tp4-c16-hbm|4|16|none"
    "tp4-c20-hbm|4|20|none"
    "tp4-c24-hbm|4|24|none"
    "tp4-c28-hbm|4|28|none"
    "tp4-c32-hbm|4|32|none"
    "tp4-c40-hbm|4|40|none"
    "tp4-c40-hicache|4|40|dram"
    "tp4-c48-hicache|4|48|dram"
    "tp4-c56-hicache|4|56|dram"
    "tp4-c64-hicache|4|64|dram"
)

# All TP2 points plus representative TP4 latency, knee, and saturation points.
# At the observed 78 minutes/point this is about 20.8 hours.
declare -ar ONE_BOOKING_POINT_IDS=(
    "tp2-c1-hbm"
    "tp2-c4-hbm"
    "tp2-c8-hbm"
    "tp2-c12-hbm"
    "tp2-c16-hbm"
    "tp2-c20-hbm"
    "tp2-c20-hicache"
    "tp2-c24-hicache"
    "tp2-c28-hicache"
    "tp2-c32-hicache"
    "tp4-c1-hbm"
    "tp4-c16-hbm"
    "tp4-c32-hbm"
    "tp4-c40-hbm"
    "tp4-c48-hicache"
    "tp4-c64-hicache"
)

point_selected() {
    local point_id="$1"
    local selected_id
    if [[ "$MODE" == "full" ]]; then
        return 0
    fi
    for selected_id in "${ONE_BOOKING_POINT_IDS[@]}"; do
        [[ "$selected_id" == "$point_id" ]] && return 0
    done
    return 1
}

git_revision() {
    git -c "safe.directory=$1" -C "$1" rev-parse HEAD 2>/dev/null || true
}

git_checkout_clean() {
    [[ -z "$(git -c "safe.directory=$1" -C "$1" status --porcelain 2>/dev/null)" ]]
}

valid_inferencex() {
    [[ -f "$1/benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh" ]]
}

resolve_inferencex() {
    local candidate revision cache_dir partial
    if [[ -n "${SWEEP_INFERENCEX_ROOT:-}" ]]; then
        candidate="$(realpath "$SWEEP_INFERENCEX_ROOT")"
        valid_inferencex "$candidate" ||
            die "not an InferenceX checkout: $candidate"
        revision="$(git_revision "$candidate")"
        [[ "$revision" == "$PINNED_INFERENCEX_REVISION" ]] ||
            die "InferenceX must be at $PINNED_INFERENCEX_REVISION, got ${revision:-unknown}"
        git_checkout_clean "$candidate" ||
            die "InferenceX checkout has local changes: $candidate"
        printf '%s\n' "$candidate"
        return
    fi

    for candidate in \
        "$(cd "$REPO_ROOT/.." && pwd)/InferenceX" \
        "${DATA_ROOT}/InferenceX" \
        "${DATA_ROOT}/zijchen/InferenceX_main_20260908" \
        "$SCRIPT_DIR/.cache/InferenceX-${PINNED_INFERENCEX_REVISION:0:12}"; do
        if valid_inferencex "$candidate" &&
            [[ "$(git_revision "$candidate")" == "$PINNED_INFERENCEX_REVISION" ]] &&
            git_checkout_clean "$candidate"; then
            realpath "$candidate"
            return
        fi
    done

    cache_dir="$SCRIPT_DIR/.cache/InferenceX-${PINNED_INFERENCEX_REVISION:0:12}"
    if [[ "$PLAN_ONLY" -eq 1 ]]; then
        die "pinned InferenceX checkout not found (an actual run would clone it to $cache_dir)"
    fi
    echo "Pinned InferenceX checkout not found; bootstrapping $cache_dir" >&2
    mkdir -p "$(dirname "$cache_dir")"
    if [[ -e "$cache_dir" ]]; then
        mv "$cache_dir" "${cache_dir}.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    partial="${cache_dir}.partial.$$"
    rm -rf "$partial"
    mkdir -p "$partial"
    git -C "$partial" init -q
    git -C "$partial" remote add origin https://github.com/SemiAnalysisAI/InferenceX.git
    timeout --signal=TERM --kill-after="$TIMEOUT_KILL_GRACE_S" \
        "$DEPENDENCY_BOOTSTRAP_TIMEOUT_S" \
        git -C "$partial" fetch --depth 1 origin "$PINNED_INFERENCEX_REVISION"
    git -C "$partial" checkout -q --detach FETCH_HEAD
    mv "$partial" "$cache_dir"
    printf '%s\n' "$cache_dir"
}

valid_aiperf() {
    [[ -f "$1/pyproject.toml" ]]
}

resolve_aiperf() {
    local candidate revision
    if [[ -n "${SWEEP_AIPERF_ROOT:-}" ]]; then
        candidate="$(realpath "$SWEEP_AIPERF_ROOT")"
        valid_aiperf "$candidate" || die "not an AIPerf checkout: $candidate"
        revision="$(git_revision "$candidate")"
        [[ "$revision" == "$PINNED_AIPERF_REVISION" ]] ||
            die "AIPerf must be at $PINNED_AIPERF_REVISION, got ${revision:-unknown}"
        git_checkout_clean "$candidate" ||
            die "AIPerf checkout has local changes: $candidate"
        printf '%s\n' "$candidate"
        return
    fi

    for candidate in \
        "$INFERENCEX_ROOT/utils/aiperf" \
        "${DATA_ROOT}/zijchen/InferenceX_main_20260908/utils/aiperf"; do
        if valid_aiperf "$candidate" &&
            [[ "$(git_revision "$candidate")" == "$PINNED_AIPERF_REVISION" ]] &&
            git_checkout_clean "$candidate"; then
            realpath "$candidate"
            return
        fi
    done

    if [[ "$PLAN_ONLY" -eq 1 ]]; then
        die "pinned AIPerf checkout not found"
    fi
    echo "Pinned AIPerf checkout not found; initializing the InferenceX submodule" >&2
    timeout --signal=TERM --kill-after="$TIMEOUT_KILL_GRACE_S" \
        "$DEPENDENCY_BOOTSTRAP_TIMEOUT_S" \
        git -C "$INFERENCEX_ROOT" submodule update --init --depth 1 utils/aiperf >&2
    candidate="$INFERENCEX_ROOT/utils/aiperf"
    valid_aiperf "$candidate" || die "AIPerf submodule initialization failed"
    [[ "$(git_revision "$candidate")" == "$PINNED_AIPERF_REVISION" ]] ||
        die "initialized AIPerf revision does not match $PINNED_AIPERF_REVISION"
    git_checkout_clean "$candidate" ||
        die "initialized AIPerf checkout has local changes"
    printf '%s\n' "$candidate"
}

resolve_model() {
    local candidate
    local -a candidates=()
    if [[ -n "${SWEEP_MODEL_PATH:-}" ]]; then
        candidates+=("$SWEEP_MODEL_PATH")
    fi
    candidates+=(
        "$DATA_ROOT/amd/Qwen3.5-397B-A17B-MXFP4"
        "$DATA_ROOT/models/Qwen3.5-397B-A17B-MXFP4"
        "$DATA_ROOT/$CURRENT_USER/models/Qwen3.5-397B-A17B-MXFP4"
        "$DATA_ROOT/zijchen/models/Qwen3.5-397B-A17B-MXFP4-0c14667cee2c"
    )
    shopt -s nullglob
    candidates+=(
        "$DATA_ROOT"/huggingface/hub/models--amd--Qwen3.5-397B-A17B-MXFP4/snapshots/*
        "$DATA_ROOT"/*/cache/hf/hub/models--amd--Qwen3.5-397B-A17B-MXFP4/snapshots/*
        "$DATA_ROOT"/*/*/cache/hf/hub/models--amd--Qwen3.5-397B-A17B-MXFP4/snapshots/*
        "$HOME"/.cache/huggingface/hub/models--amd--Qwen3.5-397B-A17B-MXFP4/snapshots/*
    )
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate/config.json" ]]; then
            realpath "$candidate"
            return
        fi
    done
    die "Qwen3.5 MXFP4 weights were not found; use --model-path once or machine.conf"
}

resolve_traces() {
    local candidate
    local -a candidates=()
    if [[ -n "${SWEEP_TRACE_DIR:-}" ]]; then
        candidates+=("$SWEEP_TRACE_DIR")
    fi
    candidates+=(
        "$DATA_ROOT/huggingface/dataset/cc-traces-weka-062126-256k"
        "$DATA_ROOT/datasets/cc-traces-weka-062126-256k"
        "$DATA_ROOT/agentx/cc-traces-weka-062126-256k"
    )
    shopt -s nullglob
    candidates+=(
        "$DATA_ROOT"/huggingface/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*
        "$DATA_ROOT"/*/cache/hf/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*
        "$DATA_ROOT"/*/*/cache/hf/hub/datasets--semianalysisai--cc-traces-weka-062126-256k/snapshots/*
    )
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
        if [[ -s "$candidate/traces.jsonl" ]]; then
            realpath "$candidate"
            return
        fi
    done
    die "AgentX traces were not found; use --trace-dir once or machine.conf"
}

for command_name in awk git python3 realpath sha256sum timeout; do
    command -v "$command_name" >/dev/null ||
        die "required command is missing: $command_name"
done

if [[ -n "${SWEEP_DATA_ROOT:-}" ]]; then
    DATA_ROOT="$(realpath "$SWEEP_DATA_ROOT")"
else
    DATA_ROOT=""
    for candidate in /data2 /data /mnt/data; do
        if [[ -d "$candidate" ]]; then
            DATA_ROOT="$candidate"
            break
        fi
    done
    [[ -n "$DATA_ROOT" ]] || die "no data root found; use --data-root or machine.conf"
fi

INFERENCEX_ROOT="$(resolve_inferencex)"
AIPERF_HOST="$(resolve_aiperf)"
MODEL_PATH="$(resolve_model)"
MODEL_CONFIG_SHA256="$(sha256sum "$MODEL_PATH/config.json" | awk '{print $1}')"
[[ "$MODEL_CONFIG_SHA256" == "$PINNED_MODEL_CONFIG_SHA256" ]] ||
    die "model config checksum does not match $PINNED_MODEL_ID"
[[ -f "$MODEL_PATH/model.safetensors.index.json" ]] ||
    die "model shard index is missing: $MODEL_PATH/model.safetensors.index.json"
MODEL_INDEX_SHA256="$(sha256sum "$MODEL_PATH/model.safetensors.index.json" | awk '{print $1}')"
[[ "$MODEL_INDEX_SHA256" == "$PINNED_MODEL_INDEX_SHA256" ]] ||
    die "model shard index checksum does not match $PINNED_MODEL_ID"
python3 - "$MODEL_PATH" <<'PY'
import json
import sys
from pathlib import Path

model = Path(sys.argv[1])
with (model / "model.safetensors.index.json").open(encoding="utf-8") as handle:
    index = json.load(handle)
shards = sorted(set(index.get("weight_map", {}).values()))
missing = [name for name in shards if not (model / name).is_file()]
if not shards or missing:
    detail = ", ".join(missing[:3]) if missing else "empty weight_map"
    raise SystemExit(f"ERROR: incomplete model checkpoint: {detail}")
PY
AGENTX_TRACE_LOCAL_DIR="$(resolve_traces)"
TRACE_SHA256="$(sha256sum "$AGENTX_TRACE_LOCAL_DIR/traces.jsonl" | awk '{print $1}')"
[[ "$TRACE_SHA256" == "$PINNED_TRACE_SHA256" ]] ||
    die "trace dataset checksum does not match revision $PINNED_TRACE_REVISION"
IMAGE="$PINNED_IMAGE"
if [[ -n "${SWEEP_CAMPAIGN_DIR:-}" ]]; then
    default_campaign_dir="$SWEEP_CAMPAIGN_DIR"
elif [[ -d "$DATA_ROOT/$CURRENT_USER" && -w "$DATA_ROOT/$CURRENT_USER" ]]; then
    default_campaign_dir="$DATA_ROOT/$CURRENT_USER/agentx-runs/agentx-qwen35-official-mi355x-v0519-20260911"
elif [[ -d "$DATA_ROOT/models" && -w "$DATA_ROOT/models" ]]; then
    default_campaign_dir="$DATA_ROOT/models/agentx-runs/$CURRENT_USER/agentx-qwen35-official-mi355x-v0519-20260911"
else
    default_campaign_dir="$REPO_ROOT/campaigns/agentx-qwen35-official-mi355x-v0519-20260911"
fi
CAMPAIGN_DIR="$(realpath -m "$default_campaign_dir")"
HF_HOME="$(realpath -m "${SWEEP_HF_HOME:-$SCRIPT_DIR/.cache/huggingface}")"
AGENTX_SHARED_CACHE_DIR="$(realpath -m "${SWEEP_SHARED_CACHE_DIR:-$CAMPAIGN_DIR/.cache}")"
PORT="${SWEEP_PORT:-$DEFAULT_PORT}"
[[ "$PORT" =~ ^[1-9][0-9]*$ && "$PORT" -le 65535 ]] ||
    die "SWEEP_PORT must be an integer from 1 to 65535"
TIME_LIMIT_S=$((TIME_LIMIT_HOURS * SECONDS_PER_HOUR))
HARD_DEADLINE_EPOCH=$((SCRIPT_START_EPOCH + TIME_LIMIT_S))

selected_count=0
for entry in "${FULL_MATRIX[@]}"; do
    IFS='|' read -r point_id _ <<< "$entry"
    if point_selected "$point_id"; then
        selected_count=$((selected_count + 1))
    fi
done
estimated_minutes=$((selected_count * ESTIMATED_POINT_MINUTES))

print_plan() {
    echo "AgentX sweep plan"
    echo "  mode:              $MODE ($selected_count of ${#FULL_MATRIX[@]} official points)"
    echo "  hard time budget:  ${TIME_LIMIT_HOURS}h"
    echo "  observed estimate: ~$((estimated_minutes / 60))h $((estimated_minutes % 60))m"
    echo "  image:             $IMAGE"
    echo "  model:             $MODEL_PATH"
    echo "  traces:            $AGENTX_TRACE_LOCAL_DIR"
    echo "  InferenceX:        $INFERENCEX_ROOT @ ${PINNED_INFERENCEX_REVISION:0:12}"
    echo "  AIPerf:            $AIPERF_HOST @ ${PINNED_AIPERF_REVISION:0:12}"
    echo "  campaign:          $CAMPAIGN_DIR"
    echo
    echo "Selected points (sequential, fresh container per point):"
    for entry in "${FULL_MATRIX[@]}"; do
        IFS='|' read -r point_id tp conc kv_mode <<< "$entry"
        if point_selected "$point_id"; then
            printf '  %-20s TP=%s C=%s KV=%s\n' "$point_id" "$tp" "$conc" "$kv_mode"
        fi
    done
    if [[ "$MODE" == "full" ]]; then
        echo
        echo "The full matrix is ~31h at the observed rate and will normally need two bookings."
    fi
}

if [[ "$PLAN_ONLY" -eq 1 ]]; then
    print_plan
    exit 0
fi

for command_name in awk curl df docker flock python3 rocm-smi sha256sum tee timeout; do
    command -v "$command_name" >/dev/null ||
        die "required command is missing: $command_name"
done

mkdir -p "$CAMPAIGN_DIR" "$CAMPAIGN_DIR/points" "$AGENTX_SHARED_CACHE_DIR"
exec 9>"$CAMPAIGN_DIR/.sweep.lock"
flock -n 9 || die "another sweep process holds $CAMPAIGN_DIR/.sweep.lock"
exec > >(tee -a "$CAMPAIGN_DIR/sweep.log") 2>&1

ACTIVE_CONTAINER=""
cleanup_active_container() {
    local rc=$?
    trap - EXIT INT TERM
    if [[ -n "$ACTIVE_CONTAINER" ]] &&
        docker inspect "$ACTIVE_CONTAINER" >/dev/null 2>&1; then
        echo "Emergency cleanup: $ACTIVE_CONTAINER"
        docker stop -t "$CONTAINER_STOP_TIMEOUT_S" "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
        docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    fi
    exit "$rc"
}
trap cleanup_active_container EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo
echo "=== AgentX sweep started $(date -Is) ==="
print_plan

timeout "$PREFLIGHT_COMMAND_TIMEOUT_S" docker info >/dev/null 2>&1 ||
    die "Docker is unavailable to this user"
[[ -e /dev/kfd && -d /dev/dri ]] ||
    die "ROCm devices /dev/kfd and /dev/dri are required"
shopt -s nullglob
render_devices=(/dev/dri/renderD*)
shopt -u nullglob
(( ${#render_devices[@]} >= 8 )) ||
    die "expected an 8-GPU MI355X node, found ${#render_devices[@]} render devices"

minimum_mem_kb=$((MIN_NODE_DRAM_MIB * 1024))
mem_total_kb="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
(( mem_total_kb >= minimum_mem_kb )) ||
    die "official HiCache budgets require at least ${MIN_NODE_DRAM_MIB} MiB host DRAM"
mem_available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( mem_available_kb >= MIN_AVAILABLE_NODE_DRAM_KB )) ||
    die "less than 1.5 TB host DRAM is currently available; stop other workloads first"

# Remove only containers owned by a prior interrupted invocation of this script.
while IFS= read -r stale_container; do
    [[ -n "$stale_container" ]] || continue
    echo "Removing stale sweep container: $stale_container"
    docker rm -f "$stale_container" >/dev/null
done < <(docker ps -a --filter 'name=^/agentx-official-q35-' --format '{{.Names}}')

gpu_memory_state="$(timeout "$PREFLIGHT_COMMAND_TIMEOUT_S" rocm-smi --showmemuse 2>/dev/null)" ||
    die "could not query ROCm GPU memory use"
busy_gpu_memory="$(awk -v max_idle_vram="$MAX_IDLE_VRAM_PERCENT" '
    /GPU Memory Allocated \(VRAM%\):/ && $NF > max_idle_vram {
        printf "%s=%s%% ", $1, $NF
    }
' <<< "$gpu_memory_state")"
[[ -z "$busy_gpu_memory" ]] ||
    die "the node is not idle ($busy_gpu_memory); wait for every GPU to release VRAM"

if command -v fuser >/dev/null; then
    busy_pids="$(fuser /dev/kfd 2>/dev/null || true)"
    [[ -z "${busy_pids//[[:space:]]/}" ]] ||
        die "/dev/kfd is already in use by PID(s): $busy_pids"
fi

python3 - "$PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket() as sock:
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"ERROR: port {port} is unavailable: {exc}")
PY

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    [[ "$PULL_IMAGE" -eq 1 ]] ||
        die "Docker image is missing and --no-pull was set: $IMAGE"
    echo "Pulling official image: $IMAGE"
    timeout --signal=TERM --kill-after="$TIMEOUT_KILL_GRACE_S" \
        "$IMAGE_PULL_TIMEOUT_S" docker pull "$IMAGE"
fi
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"

available_campaign_kb="$(df -Pk "$CAMPAIGN_DIR" | awk 'NR == 2 {print $4}')"
minimum_campaign_kb=$((MIN_CAMPAIGN_FREE_GIB * 1024 * 1024))
(( available_campaign_kb >= minimum_campaign_kb )) ||
    die "campaign filesystem needs at least ${MIN_CAMPAIGN_FREE_GIB} GiB free; configure SWEEP_CAMPAIGN_DIR on larger storage"

recipe_path="$INFERENCEX_ROOT/benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh"
recipe_sha="$(sha256sum "$recipe_path" | awk '{print $1}')"
benchmark_lib_sha="$(sha256sum "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh" | awk '{print $1}')"
CONFIG_BODY="$(cat <<EOF
schema=agentx-qwen35-mi355x-sweep-v1
image=$IMAGE
image_id=$IMAGE_ID
model=$PINNED_MODEL_ID
model_revision=$PINNED_MODEL_REVISION
model_path=$MODEL_PATH
model_config_sha256=$MODEL_CONFIG_SHA256
model_index_sha256=$MODEL_INDEX_SHA256
trace_revision=$PINNED_TRACE_REVISION
trace_sha256=$TRACE_SHA256
trace_dir=$AGENTX_TRACE_LOCAL_DIR
inferencex_revision=$PINNED_INFERENCEX_REVISION
aiperf_revision=$PINNED_AIPERF_REVISION
recipe_sha256=$recipe_sha
benchmark_lib_sha256=$benchmark_lib_sha
duration=$CANONICAL_DURATION_S
warmup_requests_per_lane=$CANONICAL_WARMUP_REQUESTS_PER_LANE
ep=1
synthetic_acceptance_length=$CANONICAL_SYNTHETIC_ACCEPTANCE_LENGTH
tp2_hicache_dram_gb=$TP2_HICACHE_DRAM_GB
tp4_hicache_dram_gb=$TP4_HICACHE_DRAM_GB
EOF
)"
CONFIG_FINGERPRINT="$(printf '%s' "$CONFIG_BODY" | sha256sum | awk '{print $1}')"

if [[ -f "$CAMPAIGN_DIR/config.fingerprint" ]]; then
    old_fingerprint="$(tr -d '[:space:]' < "$CAMPAIGN_DIR/config.fingerprint")"
    [[ "$old_fingerprint" == "$CONFIG_FINGERPRINT" ]] ||
        die "campaign configuration changed; choose a new --campaign-dir"
fi
printf '%s\n' "$CONFIG_BODY" > "$CAMPAIGN_DIR/campaign_config.txt"
printf '%s\n' "$CONFIG_FINGERPRINT" > "$CAMPAIGN_DIR/config.fingerprint"

{
    printf 'point_id\ttp\tconc\tkv_mode\tselected_in_last_run\n'
    for entry in "${FULL_MATRIX[@]}"; do
        IFS='|' read -r point_id tp conc kv_mode <<< "$entry"
        selected="no"
        point_selected "$point_id" && selected="yes"
        printf '%s\t%s\t%s\t%s\t%s\n' \
            "$point_id" "$tp" "$conc" "$kv_mode" "$selected"
    done
} > "$CAMPAIGN_DIR/manifest.tsv"

if [[ ! -f "$CAMPAIGN_DIR/status.tsv" ]]; then
    printf 'timestamp\tpoint_id\tstatus\tdetail\n' > "$CAMPAIGN_DIR/status.tsv"
fi

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    python3 "$SCRIPT_DIR/summarize_sweep.py" "$CAMPAIGN_DIR"
    echo "Preflight passed. No benchmark point was started."
    exit 0
fi

record_status() {
    local point_id="$1"
    local status="$2"
    local detail="${3:-}"
    detail="${detail//$'\t'/ }"
    detail="${detail//$'\n'/ }"
    printf '%s\t%s\t%s\t%s\n' \
        "$(date -Is)" "$point_id" "$status" "$detail" >> "$CAMPAIGN_DIR/status.tsv"
}

validate_result() {
    local result_path="$1"
    local expected_tp="$2"
    local expected_conc="$3"
    local expected_kv="$4"
    python3 - "$result_path" "$expected_tp" "$expected_conc" "$expected_kv" \
        "$CANONICAL_FAILED_REQUEST_THRESHOLD" <<'PY'
import json
import sys

path, expected_tp, expected_conc, expected_kv, failed_request_threshold = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)

throughput = (
    result.get("request_metrics", {})
    .get("throughput", {})
    .get("per_gpu", {})
    .get("total_tput_tps")
)
errors = result.get("request_accounting", {}).get("records_error_dropped")
successful = result.get("num_requests_successful")
profiled_requests = (
    successful + errors
    if isinstance(successful, int) and isinstance(errors, int)
    else 0
)
error_rate = errors / profiled_requests if profiled_requests > 0 else 1.0
valid = (
    result.get("tp") == int(expected_tp)
    and result.get("conc") == int(expected_conc)
    and result.get("kv_offloading") == expected_kv
    and isinstance(successful, int)
    and successful > 0
    and isinstance(errors, int)
    and errors >= 0
    and error_rate <= float(failed_request_threshold)
    and isinstance(throughput, (int, float))
    and throughput > 0
)
raise SystemExit(0 if valid else 1)
PY
}

write_point_env() {
    local destination="$1"
    {
        echo "# Exact clean benchmark configuration; safe to source for inspection."
        for variable_name in \
            IMAGE RECIPE_FINGERPRINT DATA_ROOT INFERENCEX_ROOT AIPERF_HOST \
            MODEL MODEL_PATH MODEL_PREFIX SERVED_MODEL_NAME FRAMEWORK PRECISION \
            SPEC_DECODING RUNNER_TYPE AGENTX_TRACE_LOCAL_DIR \
            AGENTX_TRACE_REVISION HF_HOME AGENTX_SHARED_CACHE_DIR TP EP_SIZE \
            DP_ATTENTION CONC GPUS KV_OFFLOADING KV_OFFLOAD_BACKEND \
            KV_OFFLOAD_BACKEND_METADATA TOTAL_CPU_DRAM_GB DURATION PORT \
            RESULT_FILENAME WEKA_LOADER_OVERRIDE SCHEDULER_RECV_INTERVAL \
            AIPERF_WARMUP_REQUESTS_PER_LANE AIPERF_UNSAFE_OVERRIDE \
            AIPERF_EXPERIMENTAL_FAST AIPERF_TRACE_IDLE_GAP_CAP_SECONDS \
            AIPERF_FAILED_REQUEST_THRESHOLD \
            AIPERF_LIVE_FAILED_REQUEST_THRESHOLD \
            AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES \
            AGENTIC_WARMUP_GRACE_PERIOD ENABLE_AGENTX_POWER EVAL_ONLY \
            HICACHE_RATIO HICACHE_WRITE_POLICY HICACHE_IO_BACKEND \
            HICACHE_MEM_LAYOUT SERVER_READY_TIMEOUT_S SGLANG_ROOT \
            CONTAINER_NAME RESULT_DIR; do
            printf 'export %s=%q\n' "$variable_name" "${!variable_name}"
        done
    } > "$destination"
}

OVERALL_RC=0
CONSECUTIVE_FAILURES=0
COMPLETED_THIS_RUN=0
PAUSED_FOR_DEADLINE=0

for entry in "${FULL_MATRIX[@]}"; do
    IFS='|' read -r point_id tp conc kv_mode <<< "$entry"
    point_selected "$point_id" || continue

    point_dir="$CAMPAIGN_DIR/points/$point_id"
    result_path="$point_dir/agentx_result.json"

    if [[ -f "$result_path" ]] &&
        validate_result "$result_path" "$tp" "$conc" "$kv_mode"; then
        if [[ ! -f "$point_dir/SUCCESS" ]]; then
            {
                echo "fingerprint=$CONFIG_FINGERPRINT"
                echo "recovered_at=$(date -Is)"
            } > "$point_dir/SUCCESS"
        fi
        echo "SKIP $point_id: validated result already exists"
        record_status "$point_id" "SKIPPED" "validated checkpoint"
        continue
    fi

    now_epoch=$(date +%s)
    remaining_s=$((HARD_DEADLINE_EPOCH - now_epoch))
    if (( remaining_s < POINT_START_BUDGET_S )); then
        echo "PAUSE: ${remaining_s}s remain, below the ${POINT_START_BUDGET_S}s point-start budget"
        record_status "$point_id" "PAUSED" "deadline guard before point start"
        PAUSED_FOR_DEADLINE=1
        break
    fi

    if [[ -d "$point_dir" ]]; then
        archive_dir="${point_dir}.failed-$(date -u +%Y%m%dT%H%M%SZ)"
        echo "Archiving incomplete attempt: $point_dir -> $archive_dir"
        mv "$point_dir" "$archive_dir"
    fi
    mkdir -p "$point_dir"

    if [[ "$tp" == "2" ]]; then
        GPUS="0,1"
        dram_gb="$TP2_HICACHE_DRAM_GB"
    else
        GPUS="0,1,2,3"
        dram_gb="$TP4_HICACHE_DRAM_GB"
    fi
    if [[ "$kv_mode" == "dram" ]]; then
        KV_OFFLOADING="dram"
        KV_OFFLOAD_BACKEND="hicache"
        KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
        TOTAL_CPU_DRAM_GB="$dram_gb"
    else
        KV_OFFLOADING="none"
        KV_OFFLOAD_BACKEND=""
        KV_OFFLOAD_BACKEND_METADATA=""
        TOTAL_CPU_DRAM_GB=0
    fi

    export IMAGE RECIPE_FINGERPRINT="$recipe_sha"
    export DATA_ROOT INFERENCEX_ROOT AIPERF_HOST
    export MODEL="$PINNED_MODEL_ID" MODEL_PATH MODEL_PREFIX=qwen3.5
    export SERVED_MODEL_NAME="$PINNED_MODEL_ID"
    export AGENTX_TRACE_LOCAL_DIR AGENTX_TRACE_REVISION="$PINNED_TRACE_REVISION"
    export HF_HOME AGENTX_SHARED_CACHE_DIR
    export FRAMEWORK=sglang PRECISION=fp4 SPEC_DECODING=mtp
    export RUNNER_TYPE=cluster:mi355x-amds DP_ATTENTION=false
    export TP="$tp" EP_SIZE=1 CONC="$conc" GPUS
    export KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA
    export TOTAL_CPU_DRAM_GB
    export DURATION="$CANONICAL_DURATION_S" PORT
    export CONTAINER_NAME="agentx-official-q35-$point_id"
    export RESULT_DIR="$point_dir"
    export RESULT_FILENAME=agentx_result
    export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126_256k
    export SCHEDULER_RECV_INTERVAL="$CANONICAL_SCHEDULER_RECV_INTERVAL"
    export AIPERF_WARMUP_REQUESTS_PER_LANE="$CANONICAL_WARMUP_REQUESTS_PER_LANE"
    export AIPERF_UNSAFE_OVERRIDE=false
    export AIPERF_EXPERIMENTAL_FAST=0
    export AIPERF_TRACE_IDLE_GAP_CAP_SECONDS="$CANONICAL_TRACE_IDLE_GAP_CAP_S"
    export AIPERF_FAILED_REQUEST_THRESHOLD="$CANONICAL_FAILED_REQUEST_THRESHOLD"
    export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD="$CANONICAL_FAILED_REQUEST_THRESHOLD"
    export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
    export AGENTIC_WARMUP_GRACE_PERIOD="$CANONICAL_WARMUP_GRACE_S"
    export ENABLE_AGENTX_POWER=0
    export ENABLE_TORCH_PROFILER=0
    export SERVER_READY_TIMEOUT_S="$CANONICAL_SERVER_READY_TIMEOUT_S"
    export EVAL_ONLY=false
    export HICACHE_RATIO="$CANONICAL_HICACHE_RATIO"
    export HICACHE_WRITE_POLICY="$CANONICAL_HICACHE_WRITE_POLICY"
    export HICACHE_IO_BACKEND="$CANONICAL_HICACHE_IO_BACKEND"
    export HICACHE_MEM_LAYOUT="$CANONICAL_HICACHE_MEM_LAYOUT"
    export SGLANG_ROOT=
    unset AIPERF_EXTRA_INPUTS HIP_VISIBLE_DEVICES MAX_MODEL_LEN PROFILE REQUIRE_POWER

    write_point_env "$point_dir/point.env"
    record_status "$point_id" "RUNNING" "tp=$tp conc=$conc kv=$kv_mode"
    echo
    echo "=== RUN $point_id (TP=$tp C=$conc KV=$kv_mode) $(date -Is) ==="

    ACTIVE_CONTAINER="$CONTAINER_NAME"
    now_epoch=$(date +%s)
    point_timeout_s=$((HARD_DEADLINE_EPOCH - now_epoch - CLEANUP_RESERVE_S))
    set +e
    timeout --signal=TERM --kill-after="$POINT_KILL_GRACE_S" \
        "$point_timeout_s" "$SCRIPT_DIR/run_clean_point.sh"
    point_rc=$?
    set -e

    if docker inspect "$ACTIVE_CONTAINER" >/dev/null 2>&1; then
        docker stop -t "$CONTAINER_STOP_TIMEOUT_S" "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
        docker rm -f "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    fi
    ACTIVE_CONTAINER=""

    if [[ -f "$result_path" ]] &&
        validate_result "$result_path" "$tp" "$conc" "$kv_mode"; then
        {
            echo "fingerprint=$CONFIG_FINGERPRINT"
            echo "completed_at=$(date -Is)"
            echo "point_rc=$point_rc"
        } > "$point_dir/SUCCESS"
        echo "PASS $point_id"
        record_status "$point_id" "COMPLETE" "rc=$point_rc"
        COMPLETED_THIS_RUN=$((COMPLETED_THIS_RUN + 1))
        CONSECUTIVE_FAILURES=0
    else
        echo "FAIL $point_id: rc=$point_rc; artifacts preserved in $point_dir" >&2
        record_status "$point_id" "FAILED" "rc=$point_rc"
        OVERALL_RC=1
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
    fi

    python3 "$SCRIPT_DIR/summarize_sweep.py" "$CAMPAIGN_DIR" || true

    if [[ "$point_rc" -eq 124 || "$point_rc" -eq 137 ]]; then
        echo "Sweep point reached the hard deadline; stopping after cleanup."
        PAUSED_FOR_DEADLINE=1
        break
    fi
    if (( CONSECUTIVE_FAILURES >= 2 )); then
        echo "Stopping after two consecutive failures to protect the reservation." >&2
        break
    fi
done

python3 "$SCRIPT_DIR/summarize_sweep.py" "$CAMPAIGN_DIR" || true

completed_selected=0
for entry in "${FULL_MATRIX[@]}"; do
    IFS='|' read -r point_id _ <<< "$entry"
    point_selected "$point_id" || continue
    [[ -f "$CAMPAIGN_DIR/points/$point_id/SUCCESS" ]] &&
        completed_selected=$((completed_selected + 1))
done

echo
echo "=== AgentX sweep stopped $(date -Is) ==="
echo "Completed this run: $COMPLETED_THIS_RUN"
echo "Selected progress:  $completed_selected/$selected_count"
echo "CSV summary:        $CAMPAIGN_DIR/summary.csv"
echo "JSON summary:       $CAMPAIGN_DIR/summary.json"
if (( completed_selected < selected_count )); then
    echo "Resume with the same command; validated points will be skipped."
fi
if [[ "$PAUSED_FOR_DEADLINE" -eq 1 && "$OVERALL_RC" -eq 0 ]]; then
    echo "Deadline pause was clean; no benchmark failure was recorded."
fi

exit "$OVERALL_RC"
