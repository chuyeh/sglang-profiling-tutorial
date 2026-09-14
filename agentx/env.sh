# Shared defaults for the local Qwen3.5 MXFP4 MI355X AgentX split.
# Source from the launch and client scripts; do not execute this file.

AGENTX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$AGENTX_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/data2}"

# InferenceX utils/aiperf is an empty submodule in the workspace checkout.
# Use the local tree that is already at the same pin (754356e9) rather than
# mutating the InferenceX checkout. On another machine, initialize that
# submodule and AIPERF_HOST will default to it.
INFERENCEX_ROOT="${INFERENCEX_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/InferenceX}"
if [[ -f "$INFERENCEX_ROOT/utils/aiperf/pyproject.toml" ]]; then
    _default_aiperf_host="$INFERENCEX_ROOT/utils/aiperf"
else
    _default_aiperf_host="$DATA_ROOT/zijchen/InferenceX_main_20260908/utils/aiperf"
fi
AIPERF_HOST="${AIPERF_HOST:-$_default_aiperf_host}"
unset _default_aiperf_host

IMAGE="${IMAGE:-rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260911}"
RECIPE_FINGERPRINT="${RECIPE_FINGERPRINT:-}"
CONTAINER_NAME="${CONTAINER_NAME:-agentx-qwen35-fp4-mi355x}"

MODEL="${MODEL:-amd/Qwen3.5-397B-A17B-MXFP4}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/amd/Qwen3.5-397B-A17B-MXFP4}"
MODEL_PREFIX="${MODEL_PREFIX:-qwen3.5}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
FRAMEWORK="${FRAMEWORK:-sglang}"
PRECISION="${PRECISION:-fp4}"
SPEC_DECODING="${SPEC_DECODING:-mtp}"
RUNNER_TYPE="${RUNNER_TYPE:-cluster:mi355x-amds}"

# Hub cache under /data2/huggingface/hub only has refs/main. The complete
# snapshot lives at this local-dir path; agentx_prepare_hf_cache reconstructs
# a writable hub layout that aiperf/huggingface_hub can consume.
AGENTX_TRACE_LOCAL_DIR="${AGENTX_TRACE_LOCAL_DIR:-$DATA_ROOT/huggingface/dataset/cc-traces-weka-062126-256k}"
AGENTX_TRACE_REVISION="${AGENTX_TRACE_REVISION:-8fecd2fc56694469f758f0afbbb6335ad3043740}"
HF_HOME="${HF_HOME:-$AGENTX_DIR/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
# Reuse downloads and AIPerf's uv package cache across fresh point containers.
# The virtualenv itself remains point-local so interrupted runs cannot poison
# later points.
AGENTX_SHARED_CACHE_DIR="${AGENTX_SHARED_CACHE_DIR:-$AGENTX_DIR/.cache/runtime}"

TP="${TP:-2}"
EP_SIZE="${EP_SIZE:-1}"
DP_ATTENTION="${DP_ATTENTION:-false}"
CONC="${CONC:-1}"
KV_OFFLOADING="${KV_OFFLOADING:-none}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
# InferenceX aggregation requires JSON metadata matching the yaml
# kv-offload-backend object, e.g. {"name":"hicache"}.
if [[ "$KV_OFFLOADING" == "dram" && -n "$KV_OFFLOAD_BACKEND" && "$KV_OFFLOAD_BACKEND" != "none" ]]; then
    if [[ -z "${KV_OFFLOAD_BACKEND_METADATA:-}" ]]; then
        KV_OFFLOAD_BACKEND_METADATA="{\"name\":\"$KV_OFFLOAD_BACKEND\"}"
    fi
else
    KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-}"
fi
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-1}"
DURATION="${DURATION:-180}"
PORT="${PORT:-8888}"
GPUS="${GPUS:-0,1}"
# After ROCR_VISIBLE_DEVICES filters the physical GPUs, HIP sees them as 0..N-1.
if [[ -z "${HIP_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _AGENTX_GPUS <<< "$GPUS"
    _hip_ids=()
    for ((i = 0; i < ${#_AGENTX_GPUS[@]}; i++)); do
        _hip_ids+=("$i")
    done
    HIP_VISIBLE_DEVICES="$(IFS=,; echo "${_hip_ids[*]}")"
    unset _AGENTX_GPUS _hip_ids
fi
SCHEDULER_RECV_INTERVAL="${SCHEDULER_RECV_INTERVAL:-30}"

EVAL_ONLY="${EVAL_ONLY:-false}"
ENABLE_AGENTX_POWER="${ENABLE_AGENTX_POWER:-0}"
AIPERF_UNSAFE_OVERRIDE="${AIPERF_UNSAFE_OVERRIDE:-true}"
AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"
WEKA_LOADER_OVERRIDE="${WEKA_LOADER_OVERRIDE:-semianalysis_cc_traces_weka_062126_256k}"
RESULT_FILENAME="${RESULT_FILENAME:-agentx_result}"
SERVER_READY_TIMEOUT_S="${SERVER_READY_TIMEOUT_S:-1200}"

RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/campaigns/agentx-local-smoke-$(date +%Y%m%d)}"

# Optional short torch-profiler capture during AIPerf's measured phase.
# qwen3.5_fp4_mi355x_run_client.sh starts the host-side trigger when enabled.
ENABLE_TORCH_PROFILER="${ENABLE_TORCH_PROFILER:-0}"
TORCH_PROFILE_SUBDIR="${TORCH_PROFILE_SUBDIR:-profile}"
PROFILE_DELAY_S="${PROFILE_DELAY_S:-60}"
PROFILE_WINDOW_S="${PROFILE_WINDOW_S:-12}"
PROFILE_TRIGGER_TIMEOUT_S="${PROFILE_TRIGGER_TIMEOUT_S:-3600}"
PROFILE_ACTIVE_TIMEOUT_S="${PROFILE_ACTIVE_TIMEOUT_S:-600}"
PROFILE_FLUSH_TIMEOUT_S="${PROFILE_FLUSH_TIMEOUT_S:-300}"
PROFILE_MIN_RUNNING_REQUESTS="${PROFILE_MIN_RUNNING_REQUESTS:-1}"
PROFILE_ACTIVITIES="${PROFILE_ACTIVITIES:-CPU,GPU}"
PROFILE_WITH_STACK="${PROFILE_WITH_STACK:-1}"
PROFILE_RECORD_SHAPES="${PROFILE_RECORD_SHAPES:-1}"
PROFILE_PHASE_MARKER="${PROFILE_PHASE_MARKER:-Phase profiling (profiling) started}"

# Optional: set SGLANG_ROOT to overlay a local SGLang checkout as /experiment.
SGLANG_ROOT="${SGLANG_ROOT:-}"

agentx_bool_enabled() {
    case "${1,,}" in
        1|true|yes|on) return 0 ;;
        *) return 1 ;;
    esac
}

agentx_validate_gpu_selection() {
    local gpu
    local -a selected_gpus
    local -A seen_gpus=()

    [[ "$TP" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: TP must be a positive integer, got: $TP" >&2
        return 1
    }
    IFS=',' read -ra selected_gpus <<< "$GPUS"
    if [[ "${#selected_gpus[@]}" -ne "$TP" ]]; then
        echo "ERROR: GPUS contains ${#selected_gpus[@]} devices, but TP=$TP (these must match)" >&2
        return 1
    fi
    for gpu in "${selected_gpus[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || {
            echo "ERROR: invalid GPU id in GPUS: $gpu" >&2
            return 1
        }
        [[ -z "${seen_gpus[$gpu]:-}" ]] || {
            echo "ERROR: duplicate GPU id in GPUS: $gpu" >&2
            return 1
        }
        seen_gpus["$gpu"]=1
    done
}

agentx_validate_kv_offload() {
    if [[ "$KV_OFFLOADING" == "none" ]]; then
        if [[ -n "$KV_OFFLOAD_BACKEND" || -n "$KV_OFFLOAD_BACKEND_METADATA" ]]; then
            echo "ERROR: KV offload backend metadata must be empty when KV_OFFLOADING=none" >&2
            return 1
        fi
        return 0
    fi
    if [[ "$KV_OFFLOADING" != "dram" ]]; then
        echo "ERROR: KV_OFFLOADING must be none or dram, got: $KV_OFFLOADING" >&2
        return 1
    fi
    if [[ -z "$KV_OFFLOAD_BACKEND" || "$KV_OFFLOAD_BACKEND" == "none" ]]; then
        echo "ERROR: KV_OFFLOAD_BACKEND is required when KV_OFFLOADING=dram" >&2
        return 1
    fi

    python3 - "$KV_OFFLOAD_BACKEND" "$KV_OFFLOAD_BACKEND_METADATA" <<'PY'
import json
import sys

backend, raw_metadata = sys.argv[1:]
try:
    metadata = json.loads(raw_metadata)
except json.JSONDecodeError as exc:
    raise SystemExit(f"ERROR: KV_OFFLOAD_BACKEND_METADATA must contain valid JSON: {exc}")
if not isinstance(metadata, dict) or set(metadata) not in (
    {"name"},
    {"name", "version"},
):
    raise SystemExit(
        "ERROR: KV_OFFLOAD_BACKEND_METADATA must contain 'name' and optional 'version'"
    )
if not all(isinstance(value, str) and value for value in metadata.values()):
    raise SystemExit("ERROR: KV_OFFLOAD_BACKEND_METADATA values must be non-empty strings")
if metadata["name"] != backend:
    raise SystemExit(
        "ERROR: KV_OFFLOAD_BACKEND must match KV_OFFLOAD_BACKEND_METADATA.name"
    )
PY
}

agentx_prepare_hf_cache() {
    local dest snap name ref ref_tmp current_ref
    if [[ ! -s "$AGENTX_TRACE_LOCAL_DIR/traces.jsonl" ]]; then
        echo "ERROR: missing traces at $AGENTX_TRACE_LOCAL_DIR/traces.jsonl" >&2
        return 1
    fi
    dest="$HF_HUB_CACHE/datasets--semianalysisai--cc-traces-weka-062126-256k"
    snap="$dest/snapshots/$AGENTX_TRACE_REVISION"
    mkdir -p "$dest/refs" "$snap" "$HF_DATASETS_CACHE"
    mkdir -p "$AGENTX_SHARED_CACHE_DIR/uv/bin" \
        "$AGENTX_SHARED_CACHE_DIR/uv-cache" \
        "$AGENTX_SHARED_CACHE_DIR/aiperf-mmap" \
        "$AGENTX_SHARED_CACHE_DIR/hf-datasets"
    ref="$dest/refs/main"
    current_ref=""
    if [[ -r "$ref" ]]; then
        IFS= read -r current_ref < "$ref" || true
    fi
    if [[ "$current_ref" != "$AGENTX_TRACE_REVISION" ]]; then
        # The container may atomically replace this bind-mounted file as root.
        # Replace through its user-owned parent instead of opening it in place.
        ref_tmp="$dest/refs/.main.$$"
        printf '%s\n' "$AGENTX_TRACE_REVISION" > "$ref_tmp"
        mv -f "$ref_tmp" "$ref"
    fi
    for name in traces.jsonl README.md stats.txt .gitattributes plots; do
        if [[ -e "$AGENTX_TRACE_LOCAL_DIR/$name" ]]; then
            ln -sfn "$AGENTX_TRACE_LOCAL_DIR/$name" "$snap/$name"
        fi
    done
}

agentx_require_paths() {
    if [[ ! -d "$DATA_ROOT" ]]; then
        echo "ERROR: DATA_ROOT not found: $DATA_ROOT" >&2
        return 1
    fi
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: MODEL_PATH not found: $MODEL_PATH" >&2
        return 1
    fi
    if [[ ! -d "$INFERENCEX_ROOT/benchmarks" ]]; then
        echo "ERROR: INFERENCEX_ROOT missing benchmarks: $INFERENCEX_ROOT" >&2
        return 1
    fi
    if [[ ! -f "$AIPERF_HOST/pyproject.toml" ]]; then
        echo "ERROR: AIPERF_HOST is not an aiperf checkout: $AIPERF_HOST" >&2
        return 1
    fi
}

agentx_container_env() {
    # Echo docker -e arguments for the AgentX container. Server + client share these.
    cat <<EOF
PYTHONNOUSERSITE=1
SGLANG_USE_AITER=1
SGLANG_USE_AITER_UNIFIED_ATTN=1
AITER_FLYDSL_FORCE=1
SGLANG_MAMBA_SSM_DTYPE=bfloat16
ROCM_QUICK_REDUCE_QUANTIZATION=INT8
SGLANG_TIMEOUT_KEEP_ALIVE=1800
ROCR_VISIBLE_DEVICES=$GPUS
HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
HF_HOME=/hf_home
HF_HUB_CACHE=/hf_home/hub
HF_DATASETS_CACHE=/agentx-cache/hf-datasets
HF_HUB_DISABLE_XET=1
HF_HUB_OFFLINE=
TRANSFORMERS_OFFLINE=
TOKENIZERS_PARALLELISM=false
IMAGE=$IMAGE
RECIPE_FINGERPRINT=$RECIPE_FINGERPRINT
MODEL=$MODEL
MODEL_PATH=$MODEL_PATH
MODEL_PREFIX=$MODEL_PREFIX
SERVED_MODEL_NAME=$SERVED_MODEL_NAME
FRAMEWORK=$FRAMEWORK
PRECISION=$PRECISION
SPEC_DECODING=$SPEC_DECODING
RUNNER_TYPE=$RUNNER_TYPE
SCENARIO_TYPE=agentic-coding
IS_AGENTIC=1
IS_MULTINODE=false
TP=$TP
EP_SIZE=$EP_SIZE
DP_ATTENTION=$DP_ATTENTION
CONC=$CONC
KV_OFFLOADING=$KV_OFFLOADING
KV_OFFLOAD_BACKEND=$KV_OFFLOAD_BACKEND
KV_OFFLOAD_BACKEND_METADATA=$KV_OFFLOAD_BACKEND_METADATA
TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB
DURATION=$DURATION
PORT=$PORT
EVAL_ONLY=$EVAL_ONLY
ENABLE_AGENTX_POWER=$ENABLE_AGENTX_POWER
AIPERF_UNSAFE_OVERRIDE=$AIPERF_UNSAFE_OVERRIDE
AIPERF_WARMUP_REQUESTS_PER_LANE=${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}
WEKA_LOADER_OVERRIDE=$WEKA_LOADER_OVERRIDE
RESULT_FILENAME=$RESULT_FILENAME
RESULT_DIR=/results
AGENTIC_OUTPUT_DIR=/results
INFMAX_CONTAINER_WORKSPACE=/inferencex
AIPERF_DIR=/opt/aiperf
AIPERF_RUNTIME_DIR=/tmp/inferencex-agentic
AIPERF_UV_INSTALL_DIR=/agentx-cache/uv/bin
AIPERF_UV_CACHE_DIR=/agentx-cache/uv-cache
AIPERF_DATASET_MMAP_CACHE_DIR=/agentx-cache/aiperf-mmap
AIPERF_SERVER_METRICS_URLS=http://localhost:${PORT}/metrics
AIPERF_REQUIRED_SERVER_METRIC_PREFIX=sglang:
PYTHONPYCACHEPREFIX=/tmp/inferencex-pycache
PYTHONDONTWRITEBYTECODE=1
EOF
    if agentx_bool_enabled "$ENABLE_TORCH_PROFILER"; then
        cat <<EOF
SGLANG_TORCH_PROFILER_DIR=/results/$TORCH_PROFILE_SUBDIR
SGLANG_PROFILE_WITH_STACK=$PROFILE_WITH_STACK
SGLANG_PROFILE_RECORD_SHAPES=$PROFILE_RECORD_SHAPES
EOF
    fi
    if [[ "${EVAL_ONLY}" != "true" ]]; then
        cat <<EOF
SGLANG_SIMULATE_ACC_LEN=3.39
SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
EOF
    fi
    if [[ -n "$SGLANG_ROOT" ]]; then
        echo "PYTHONPATH=/experiment/python"
    fi
}
