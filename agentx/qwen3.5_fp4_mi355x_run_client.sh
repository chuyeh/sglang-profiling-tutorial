#!/usr/bin/env bash
set -euo pipefail

# Run AgentX AIPerf trace replay against an already-launched SGLang container.
# Does not stop the server.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

agentx_validate_gpu_selection
agentx_validate_kv_offload
agentx_require_paths
agentx_prepare_hf_cache
mkdir -p "$RESULT_DIR"

if [[ -z "${CONTAINER_NAME:-}" && -f "$RESULT_DIR/container_name.txt" ]]; then
    CONTAINER_NAME="$(tr -d '[:space:]' < "$RESULT_DIR/container_name.txt")"
fi

if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null | grep -q true; then
    echo "ERROR: container $CONTAINER_NAME is not running. Launch it with qwen3.5_fp4_mi355x_launch_server.sh first." >&2
    exit 1
fi

if ! curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "ERROR: SGLang is not reachable at http://127.0.0.1:${PORT}/v1/models" >&2
    exit 1
fi

profile_pid=""
profile_rc=0
profile_sentinel=""
if agentx_bool_enabled "$ENABLE_TORCH_PROFILER"; then
    profile_sentinel="$RESULT_DIR/.agentx_client_running"
    touch "$profile_sentinel"
    echo "Arming torch-profiler trigger; log: $RESULT_DIR/profile_trigger.log"
    PROFILE_CLIENT_SENTINEL="$profile_sentinel" \
        "$SCRIPT_DIR/trigger_torch_profile.sh" &
    profile_pid=$!
    sleep 1
    if ! kill -0 "$profile_pid" 2>/dev/null; then
        set +e
        wait "$profile_pid"
        profile_rc=$?
        set -e
        rm -f "$profile_sentinel"
        echo "ERROR: torch-profiler trigger failed to arm (rc=$profile_rc). See $RESULT_DIR/profile_trigger.log" >&2
        exit 1
    fi
fi

EXEC_ARGS=(docker exec)
while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    EXEC_ARGS+=(-e "$line")
done < <(agentx_container_env)

echo "Running AgentX replay in $CONTAINER_NAME (CONC=$CONC DURATION=$DURATION)"
set +e
"${EXEC_ARGS[@]}" "$CONTAINER_NAME" bash /opt/agentx/run_replay_inside.sh \
    2>&1 | tee "$RESULT_DIR/client.log"
client_rc=${PIPESTATUS[0]}
set -e

if [[ -n "$profile_pid" ]]; then
    rm -f "$profile_sentinel"
    set +e
    wait "$profile_pid"
    profile_rc=$?
    set -e
    if [[ "$profile_rc" -ne 0 ]]; then
        echo "ERROR: torch-profiler capture failed (rc=$profile_rc). See $RESULT_DIR/profile_trigger.log" >&2
    else
        echo "Torch profile: $RESULT_DIR/$TORCH_PROFILE_SUBDIR"
    fi
fi

docker logs "$CONTAINER_NAME" > "$RESULT_DIR/server.log" 2>&1 || true

if [[ -d "$RESULT_DIR/aiperf_artifacts" ]]; then
    echo "AIPerf artifacts: $RESULT_DIR/aiperf_artifacts"
else
    echo "WARNING: no aiperf_artifacts directory under $RESULT_DIR" >&2
fi

if [[ -f "$RESULT_DIR/${RESULT_FILENAME}.json" ]]; then
    echo "Aggregate result: $RESULT_DIR/${RESULT_FILENAME}.json"
fi

if [[ "$client_rc" -eq 0 && "$profile_rc" -ne 0 ]]; then
    exit "$profile_rc"
fi
exit "$client_rc"
