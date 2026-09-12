#!/usr/bin/env bash
# Launch, benchmark, and tear down one clean AgentX point.
#
# The official sweep exports the point configuration before invoking this
# helper. It intentionally forces the torch profiler off.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${RESULT_DIR:?RESULT_DIR must be set}"
: "${CONTAINER_NAME:?CONTAINER_NAME must be set}"

export ENABLE_TORCH_PROFILER=0
mkdir -p "$RESULT_DIR"

cleanup() {
    local rc=$?
    trap - EXIT INT TERM
    if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
        docker logs "$CONTAINER_NAME" > "$RESULT_DIR/server.log" 2>&1 || true
        echo "Stopping $CONTAINER_NAME"
        docker stop -t 60 "$CONTAINER_NAME" >/dev/null 2>&1 || true
        docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
    fi
    exit "$rc"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$SCRIPT_DIR/qwen3.5_fp4_mi355x_launch_server.sh"
"$SCRIPT_DIR/qwen3.5_fp4_mi355x_run_client.sh"

echo "Clean AgentX point complete: $RESULT_DIR"
