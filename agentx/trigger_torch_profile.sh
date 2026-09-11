#!/usr/bin/env bash
# Host-side controller for a short SGLang torch-profiler capture.
#
# The controller waits for AIPerf's measured profiling phase, applies an
# optional delay, then waits until SGLang has active requests before calling
# /start_profile. The profile is stopped after PROFILE_WINDOW_S seconds.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "$SCRIPT_DIR/env.sh"

PROFILE_PHASE_LOG="${PROFILE_PHASE_LOG:-$RESULT_DIR/aiperf_artifacts/logs/aiperf.log}"
PROFILE_HOST_DIR="$RESULT_DIR/$TORCH_PROFILE_SUBDIR"
PROFILE_CONTAINER_DIR="/results/$TORCH_PROFILE_SUBDIR"
PROFILE_TRIGGER_LOG="${PROFILE_TRIGGER_LOG:-$RESULT_DIR/profile_trigger.log}"
PROFILE_CLIENT_SENTINEL="${PROFILE_CLIENT_SENTINEL:-}"

mkdir -p "$RESULT_DIR"
exec >>"$PROFILE_TRIGGER_LOG" 2>&1

log() {
    printf '%s %s\n' "$(date -Iseconds)" "$*"
}

die() {
    log "ERROR: $*"
    exit 1
}

require_uint() {
    local name="$1" value="$2"
    [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer, got: $value"
}

client_is_running() {
    [[ -z "$PROFILE_CLIENT_SENTINEL" || -e "$PROFILE_CLIENT_SENTINEL" ]]
}

sleep_checked() {
    local remaining="$1"
    while (( remaining > 0 )); do
        client_is_running || die "AgentX client exited before the profile capture completed"
        sleep 1
        remaining=$((remaining - 1))
    done
}

profile_active=0

stop_profile() {
    local response
    if (( profile_active == 0 )); then
        return 0
    fi
    log "Stopping torch profiler"
    if response="$(curl -fsS --max-time 300 -X POST \
        "http://127.0.0.1:${PORT}/stop_profile" 2>&1)"; then
        log "stop_profile response: ${response:-<empty>}"
        profile_active=0
        return 0
    fi
    log "ERROR: stop_profile failed: $response"
    profile_active=0
    return 1
}

finish() {
    local rc=$?
    trap - EXIT INT TERM HUP
    if (( profile_active == 1 )); then
        stop_profile || true
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 130' INT TERM HUP

require_uint PROFILE_DELAY_S "$PROFILE_DELAY_S"
require_uint PROFILE_WINDOW_S "$PROFILE_WINDOW_S"
require_uint PROFILE_TRIGGER_TIMEOUT_S "$PROFILE_TRIGGER_TIMEOUT_S"
require_uint PROFILE_ACTIVE_TIMEOUT_S "$PROFILE_ACTIVE_TIMEOUT_S"
require_uint PROFILE_FLUSH_TIMEOUT_S "$PROFILE_FLUSH_TIMEOUT_S"
require_uint PROFILE_MIN_RUNNING_REQUESTS "$PROFILE_MIN_RUNNING_REQUESTS"
require_uint DURATION "$DURATION"

(( PROFILE_WINDOW_S > 0 )) || die "PROFILE_WINDOW_S must be greater than zero"
if (( PROFILE_DELAY_S + PROFILE_WINDOW_S >= DURATION )); then
    die "PROFILE_DELAY_S + PROFILE_WINDOW_S must be less than DURATION (${PROFILE_DELAY_S} + ${PROFILE_WINDOW_S} >= ${DURATION})"
fi
if [[ "$TORCH_PROFILE_SUBDIR" == /* || "$TORCH_PROFILE_SUBDIR" == *".."* ]]; then
    die "TORCH_PROFILE_SUBDIR must be a safe relative path under RESULT_DIR"
fi

IFS=',' read -ra profile_activities <<< "$PROFILE_ACTIVITIES"
activities_json=""
for activity in "${profile_activities[@]}"; do
    activity="${activity//[[:space:]]/}"
    case "$activity" in
        CPU|GPU|MEM|RPD) ;;
        *) die "unsupported profiler activity '$activity' (use CPU,GPU,MEM,RPD)" ;;
    esac
    [[ -n "$activities_json" ]] && activities_json+=","
    activities_json+="\"$activity\""
done
[[ -n "$activities_json" ]] || die "PROFILE_ACTIVITIES cannot be empty"

if [[ -d "$PROFILE_HOST_DIR" ]] && python3 - "$PROFILE_HOST_DIR" <<'PY'
import pathlib
import sys

raise SystemExit(0 if any(pathlib.Path(sys.argv[1]).iterdir()) else 1)
PY
then
    if ! agentx_bool_enabled "${PROFILE_ALLOW_EXISTING:-0}"; then
        die "profile output is not empty: $PROFILE_HOST_DIR (use a new RESULT_DIR)"
    fi
    log "WARNING: PROFILE_ALLOW_EXISTING=1; new traces will mix with existing files"
fi
mkdir -p "$PROFILE_HOST_DIR"

log "Armed: phase_log=$PROFILE_PHASE_LOG marker='$PROFILE_PHASE_MARKER' delay=${PROFILE_DELAY_S}s window=${PROFILE_WINDOW_S}s"

# Follow only content written after this controller starts. This prevents a
# marker from an older run in a reused result directory from triggering early.
if ! python3 - "$PROFILE_PHASE_LOG" "$PROFILE_PHASE_MARKER" \
    "$PROFILE_TRIGGER_TIMEOUT_S" "$PROFILE_CLIENT_SENTINEL" <<'PY'
import os
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
marker = sys.argv[2]
timeout = int(sys.argv[3])
sentinel = pathlib.Path(sys.argv[4]) if sys.argv[4] else None
deadline = time.monotonic() + timeout

initial_identity = None
initial_offset = 0
try:
    stat = path.stat()
    initial_identity = (stat.st_dev, stat.st_ino)
    initial_offset = stat.st_size
except FileNotFoundError:
    pass

stream = None
identity = None
first_open = True
try:
    while time.monotonic() < deadline:
        if sentinel is not None and not sentinel.exists():
            print("AgentX client exited before the measured phase marker appeared", file=sys.stderr)
            raise SystemExit(3)
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(0.5)
            continue

        current_identity = (stat.st_dev, stat.st_ino)
        if stream is None or current_identity != identity:
            if stream is not None:
                stream.close()
            stream = path.open("r", encoding="utf-8", errors="replace")
            identity = current_identity
            if first_open and identity == initial_identity:
                stream.seek(initial_offset)
            first_open = False
        elif stat.st_size < stream.tell():
            stream.seek(0)

        line = stream.readline()
        if not line:
            time.sleep(0.5)
            continue
        if marker in line:
            print(line.rstrip())
            raise SystemExit(0)
finally:
    if stream is not None:
        stream.close()

print(f"Timed out after {timeout}s waiting for marker {marker!r} in {path}", file=sys.stderr)
raise SystemExit(2)
PY
then
    die "did not observe AIPerf's measured profiling phase"
fi

log "AIPerf measured phase detected"
sleep_checked "$PROFILE_DELAY_S"

running_requests() {
    local metrics
    metrics="$(curl -fsS --max-time 10 "http://127.0.0.1:${PORT}/metrics")" || return 1
    awk '
        $1 == "sglang:num_running_reqs" ||
        index($1, "sglang:num_running_reqs{") == 1 {
            total += $2
            found = 1
        }
        END {
            if (!found) exit 1
            printf "%.0f\n", total
        }
    ' <<< "$metrics"
}

active_deadline=$((SECONDS + PROFILE_ACTIVE_TIMEOUT_S))
running=0
while (( SECONDS < active_deadline )); do
    client_is_running || die "AgentX client exited while waiting for active SGLang requests"
    if running="$(running_requests)" && (( running >= PROFILE_MIN_RUNNING_REQUESTS )); then
        break
    fi
    sleep 1
done
if (( running < PROFILE_MIN_RUNNING_REQUESTS )); then
    die "no active capture window found within ${PROFILE_ACTIVE_TIMEOUT_S}s (last running requests: $running)"
fi

with_stack=false
record_shapes=false
agentx_bool_enabled "$PROFILE_WITH_STACK" && with_stack=true
agentx_bool_enabled "$PROFILE_RECORD_SHAPES" && record_shapes=true
payload="{\"output_dir\":\"$PROFILE_CONTAINER_DIR\",\"activities\":[$activities_json],\"with_stack\":$with_stack,\"record_shapes\":$record_shapes}"

log "Starting torch profiler with running_requests=$running payload=$payload"
if ! response="$(curl -fsS --max-time 120 -X POST \
    "http://127.0.0.1:${PORT}/start_profile" \
    -H "Content-Type: application/json" \
    -d "$payload" 2>&1)"; then
    die "start_profile failed: $response"
fi
profile_active=1
log "start_profile response: ${response:-<empty>}"

sleep_checked "$PROFILE_WINDOW_S"
stop_profile

flush_deadline=$((SECONDS + PROFILE_FLUSH_TIMEOUT_S))
profile_file_count=0
while (( SECONDS < flush_deadline )); do
    profile_file_count="$(python3 - "$PROFILE_HOST_DIR" <<'PY'
import pathlib
import sys

print(sum(1 for path in pathlib.Path(sys.argv[1]).rglob("*") if path.is_file()))
PY
)"
    (( profile_file_count > 0 )) && break
    sleep 1
done
(( profile_file_count > 0 )) || die "stop_profile returned but no files appeared in $PROFILE_HOST_DIR"

log "Capture complete: files=$profile_file_count output=$PROFILE_HOST_DIR"
