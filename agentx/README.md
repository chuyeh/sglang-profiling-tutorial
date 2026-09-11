# Local AgentX (Qwen3.5 MXFP4 / MI355X / SGLang MTP)

Split InferenceX’s combined AgentX recipe into a **server launch** script and an **AIPerf client** script so you can keep SGLang up, rerun the benchmark, and stop the container yourself.

This is a local workflow. It is not a drop-in replacement for an official InferenceX CI submission (those use `DURATION=3600` and the `lmsysorg/sglang-rocm` image).

Upstream recipe: [qwen3.5_fp4_mi355x_sglang_mtp.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh).

## Prerequisites

- Docker with ROCm devices (`/dev/kfd`, `/dev/dri`)
- Image `rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260906` (override with `IMAGE`)
- Weights: `/data2/amd/Qwen3.5-397B-A17B-MXFP4`
- Traces: `/data2/huggingface/dataset/cc-traces-weka-062126-256k/traces.jsonl`
- InferenceX checkout: `/home/chuyeh/workspace/InferenceX`
- AIPerf tree at pin `754356e9`: `/data2/zijchen/InferenceX_main_20260908/utils/aiperf`

Do **not** run the upstream combined recipe. Its EXIT trap kills the server after the client.

## Quick start

From this directory:

```bash
cd /home/chuyeh/workspace/sglang-profiling-tutorial/agentx

# 1) Launch SGLang. Waits until /v1/models is ready, then leaves the container up.
./qwen3.5_fp4_mi355x_launch_server.sh

# 2) Run AIPerf AgentX trace replay against that container.
./qwen3.5_fp4_mi355x_run_client.sh

# 3) Release GPUs when you are done.
docker stop agentx-qwen35-fp4-mi355x
docker rm agentx-qwen35-fp4-mi355x
```

Load usually takes a few minutes. The client installs an isolated AIPerf venv inside the container on every run.

Defaults: **TP=2**, **CONC=1**, **kv-offloading=none**, **DURATION=180**, GPUs `0,1`, port `8888`. Throughput uses golden synthetic acceptance `SGLANG_SIMULATE_ACC_LEN=3.39`.

## Layout

| File | Role |
|---|---|
| `env.sh` | Shared defaults. Source only; do not execute. |
| `qwen3.5_fp4_mi355x_launch_server.sh` | `docker run -d` with recipe serve flags |
| `qwen3.5_fp4_mi355x_run_client.sh` | `docker exec` of the AIPerf replay |
| `run_replay_inside.sh` | In-container helper (sources InferenceX `benchmark_lib.sh`) |
| `trigger_torch_profile.sh` | Phase-aware host controller for `/start_profile` and `/stop_profile` |
| `run_profile_point.sh` | One-point launch + replay + torch-profile convenience wrapper |
| `run_tp4_hicache_canonical.sh` | Canonical TP4 HiCache benchmark sweep (no torch profiler) |

## Results

Default directory: `../campaigns/agentx-local-smoke-YYYYMMDD/`

| Artifact | Contents |
|---|---|
| `agentx_result.json` | Aggregated AgentX metrics |
| `aiperf_artifacts/` | Profile export + `sglang:` server metrics |
| `client.log` / `server.log` | Client and Docker logs |
| `sglang_command.txt` | Exact `launch_server` argv |
| `metrics_plots.png` | AIPerf plots |
| `profile/` | Per-rank torch-profiler traces when profiling is enabled |
| `profile_trigger.log` | Profile phase, active-request gate, and endpoint responses |

Override with `RESULT_DIR=/path/to/run`. Use the same `RESULT_DIR` for launch and client.

## Common overrides

Set env vars **before** the script. `TP`, `CONC`, and `KV_OFFLOADING` are baked into the server process (`--max-running-requests=2*CONC`, CUDA-graph batch size, HiCache flags). Changing those requires a relaunch.

Different GPUs / port:

```bash
GPUS=2,3 PORT=8890 CONTAINER_NAME=agentx-qwen35-tp2-c1 \
  ./qwen3.5_fp4_mi355x_launch_server.sh

GPUS=2,3 PORT=8890 CONTAINER_NAME=agentx-qwen35-tp2-c1 \
  ./qwen3.5_fp4_mi355x_run_client.sh
```

Longer TP2 point (still not the official 3600s window):

```bash
CONC=8 DURATION=900 AIPERF_UNSAFE_OVERRIDE=true \
  AIPERF_WARMUP_REQUESTS_PER_LANE=10 \
  RESULT_DIR=$PWD/../campaigns/agentx-tp2-c8 \
  ./qwen3.5_fp4_mi355x_launch_server.sh

CONC=8 DURATION=900 AIPERF_UNSAFE_OVERRIDE=true \
  AIPERF_WARMUP_REQUESTS_PER_LANE=10 \
  RESULT_DIR=$PWD/../campaigns/agentx-tp2-c8 \
  ./qwen3.5_fp4_mi355x_run_client.sh
```

Durations below 900s need `AIPERF_UNSAFE_OVERRIDE=true`. InferenceX’s AgentX scenario otherwise requires at least 900s.

HiCache (relaunch the server):

```bash
KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache TOTAL_CPU_DRAM_GB=512 CONC=24 \
  ./qwen3.5_fp4_mi355x_launch_server.sh
```

TP4 needs four GPUs, for example `TP=4 GPUS=0,1,2,3`.

To overlay a local SGLang checkout instead of the image tree:

```bash
SGLANG_ROOT=/home/chuyeh/workspace/sglang ./qwen3.5_fp4_mi355x_launch_server.sh
```

## Recommended benchmark-then-profile workflow

Treat the benchmark and profiler as two passes of the same experiment:

```text
clean benchmark -> inspect results -> select a bad point -> fresh profiled replay
```

Do **not** report the metrics from the profiled replay. Torch profiling slows CPU and GPU work, and AgentX is closed-loop: slower responses change when later turns and subagents are submitted. A 12-second capture can therefore affect more than those 12 seconds of aggregate results.

A torch trace is not a flight recorder. It cannot recover work that happened before profiling was enabled. “Profile after benchmarking” means rerunning the selected `(TP, concurrency, KV mode)` point with the profiler active.

### Pass 1: run a clean benchmark

Choose one experiment identity and keep its serving configuration in exported variables. This example uses the TP4 HiCache concurrency-56 point:

```bash
cd /path/to/sglang-profiling-tutorial/agentx

CASE_ID=qwen35-tp4-c56-hicache
CASE_ROOT=$PWD/../campaigns/$CASE_ID
KV_MODE=dram

export IMAGE=rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260906
export SGLANG_ROOT=/path/to/your/sglang       # omit to use the image's SGLang
export TP=4 EP_SIZE=1 CONC=56 GPUS=4,5,6,7
export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache
export TOTAL_CPU_DRAM_GB=1199
export DURATION=3600
export AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_UNSAFE_OVERRIDE=false
export ENABLE_TORCH_PROFILER=0
export PORT=18988
export CONTAINER_NAME=agentx-$CASE_ID-benchmark
export RESULT_DIR=$CASE_ROOT/benchmark

./qwen3.5_fp4_mi355x_launch_server.sh
./qwen3.5_fp4_mi355x_run_client.sh
docker stop "$CONTAINER_NAME"
docker rm "$CONTAINER_NAME"
```

Use `DURATION=3600` for an InferenceX-style result. A shorter duration of at least 900 seconds is useful for private triage but is not directly comparable to the canonical one-hour result.

### Pass 2: decide whether the point needs profiling

Inspect the clean aggregate:

```bash
jq '{
  conc,
  successful_requests: .num_requests_successful,
  errors: .request_accounting.records_error_dropped,
  throughput_per_gpu_tps:
    .request_metrics.throughput.per_gpu.total_tput_tps,
  p90_e2e_interactivity:
    .request_metrics.latency.e2e_norm_intvty.p90,
  p90_ttft_seconds: .request_metrics.latency.ttft.p90,
  gpu_kv_usage: .server_metrics.kv_cache.gpu_usage_pct,
  cpu_kv_usage: .server_metrics.kv_cache.cpu_usage_pct
}' "$CASE_ROOT/benchmark/agentx_result.json"
```

A point is a useful profiling candidate when it has:

- lower throughput or interactivity than its baseline;
- unexpectedly high TTFT or queueing;
- request errors, timeouts, or a server stall;
- unusual GPU/CPU KV-cache behavior;
- a regression isolated to one TP, concurrency, or HiCache setting.

Compare like with like. For example, compare TP4 HiCache C56 against another TP4 HiCache C56 run, not against TP2 C28.

The clean run's `aiperf_artifacts/profile_export_aiperf_timeslices.csv` and `client.log` can help identify whether the problem appears early, late, or only under high queue/KV pressure. Use that evidence to choose `PROFILE_DELAY_S` for the replay.

### Pass 3: rerun that exact point with torch profiling

Start from a fresh container so weights, KV cache, and replay warmup begin from the same state. Preserve the image/source commit, model, dataset revision, TP/EP, concurrency, GPU count, KV mode, DRAM budget, server flags, warmup count, and synthetic-acceptance settings.

```bash
PROFILE_RESULT=$CASE_ROOT/profile

RESULT_DIR="$PROFILE_RESULT" \
CONTAINER_NAME=agentx-$CASE_ID-profile \
PORT=18988 \
DURATION=3600 \
PROFILE_DELAY_S=300 \
PROFILE_WINDOW_S=12 \
PROFILE_MIN_RUNNING_REQUESTS=1 \
STOP_CONTAINER_AFTER=1 \
  ./run_profile_point.sh "$CASE_ID-profile" "$TP" "$CONC" "$GPUS" "$KV_MODE"
```

For an HBM-only point, set `KV_MODE=none` and omit the HiCache DRAM budget. `run_profile_point.sh` accepts:

```text
run_profile_point.sh <label> <TP> <CONC> <GPUS> <none|dram>
```

Its profiler-specific defaults are:

| Variable | Default | Meaning |
|---|---:|---|
| `DURATION` | `1500` | AIPerf measured duration when not already exported |
| `PROFILE_DELAY_S` (`WARM`) | `840` | Delay **after AIPerf’s measured phase starts** |
| `PROFILE_WINDOW_S` (`WINDOW`) | `12` | Torch-profiler capture duration |
| `PROFILE_MIN_RUNNING_REQUESTS` | `1` | Do not start during an AgentX idle gap |
| `PROFILE_ACTIVITIES` | `CPU,GPU` | SGLang activities (`GPU` is correct for ROCm PyTorch) |
| `PROFILE_WITH_STACK` | `1` | Collect operator source stacks |
| `PROFILE_RECORD_SHAPES` | `1` | Collect operator input shapes |
| `STOP_CONTAINER_AFTER` | `0` | Leave the server available for inspection |

The trigger:

1. follows the new AIPerf log until `Phase profiling (profiling) started`;
2. waits `PROFILE_DELAY_S`;
3. polls `sglang:num_running_reqs` so an inter-turn idle gap is not captured;
4. calls `/start_profile`, records for `PROFILE_WINDOW_S`, then calls `/stop_profile`;
5. fails if phase detection, active-load detection, endpoint calls, or trace export fail.

If the clean run shows the issue only late in the hour, keep the same `DURATION` and move `PROFILE_DELAY_S` near that period. If the issue is steady-state and appears early, the profiling replay can be shortened, provided `PROFILE_DELAY_S + PROFILE_WINDOW_S < DURATION`.

### Pass 4: correlate the trace with the clean result

Keep these artifacts together:

```text
benchmark/agentx_result.json
benchmark/aiperf_artifacts/profile_export_aiperf_timeslices.csv
benchmark/client.log
benchmark/server.log
profile/profile_point_env.txt
profile/sglang_command.txt
profile/profile_trigger.log
profile/profile/*
```

Open each per-rank `*.trace.json` or `*.trace.json.gz` file in Perfetto. Start by checking:

- CPU gaps before GPU kernels: scheduler, tokenization, or launch overhead;
- long GPU kernels or empty GPU regions;
- TP collective overlap and rank imbalance;
- prefill-versus-decode behavior;
- HiCache copy/load-back work around latency spikes.

`profile_trigger.log` records the exact measured-phase detection time, active request count, profile payload, and start/stop responses. Use it to align the trace with `server.log` and AIPerf timeslices.

Multi-rank traces can be large. Start with a 5-12 second window. Disable `PROFILE_WITH_STACK` or `PROFILE_RECORD_SHAPES` when kernel timing is sufficient.

### One-pass mode for quick debugging

You can also enable profiling directly in the manually split launch/client workflow:

```bash
export TP=4 CONC=56 GPUS=4,5,6,7
export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache TOTAL_CPU_DRAM_GB=1199
export RESULT_DIR=$PWD/../campaigns/my-profile-tp4-c56
export ENABLE_TORCH_PROFILER=1 PROFILE_DELAY_S=300 PROFILE_WINDOW_S=12

./qwen3.5_fp4_mi355x_launch_server.sh
./qwen3.5_fp4_mi355x_run_client.sh
```

This is convenient for quick investigation, but its aggregate benchmark metrics are profiler-contaminated. Always use a fresh `RESULT_DIR` for every capture.

## Moving to another 8-GPU MI355X server

The scripts do not reserve GPUs 4-7; any physical GPU list is accepted. The number of IDs must equal `TP`. For example, use `GPUS=0,1,2,3` or `GPUS=4,5,6,7` for TP4. The official Qwen3.5 AgentX matrix uses TP2 and TP4; having eight GPUs available does not by itself make TP8 comparable to those published points.

Set machine-specific paths before launch:

```bash
export DATA_ROOT=/data2
export INFERENCEX_ROOT=/path/to/InferenceX
export AIPERF_HOST=/path/to/InferenceX/utils/aiperf
export MODEL_PATH=/data2/models/Qwen3.5-397B-A17B-MXFP4
export AGENTX_TRACE_LOCAL_DIR=/data2/datasets/cc-traces-weka-062126-256k
export SGLANG_ROOT=/path/to/your/sglang       # optional source overlay
export HF_HOME=/data2/cache/agentx-huggingface
```

`AIPERF_HOST` must contain `pyproject.toml` and should be the InferenceX-pinned AIPerf checkout. `DATA_ROOT` is mounted read-only at the same absolute path inside the container, so keep the model and trace paths under it. Run profile points sequentially unless you intentionally assign disjoint GPUs, result directories, ports, and host-DRAM budgets.

## Defaults worth knowing

| Variable | Default |
|---|---|
| `IMAGE` | `rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260906` |
| `MODEL` / `MODEL_PATH` | `amd/Qwen3.5-397B-A17B-MXFP4` / `/data2/amd/Qwen3.5-397B-A17B-MXFP4` |
| `TP` / `EP_SIZE` / `CONC` | `2` / `1` / `1` |
| `KV_OFFLOADING` | `none` |
| `DURATION` | `180` |
| `AIPERF_WARMUP_REQUESTS_PER_LANE` | `1` |
| `ENABLE_AGENTX_POWER` | `0` |
| `INFERENCEX_ROOT` | `/home/chuyeh/workspace/InferenceX` |
| `AIPERF_HOST` | `/data2/zijchen/InferenceX_main_20260908/utils/aiperf` |

InferenceX TP2 conc list is `[1, 4, 8, 12, 16, 20]`. TP4 is `[1, 4, 8, 12, 16, 20, 24, 28, 32, 40]`. HiCache arms sit at higher conc. Safest is one container per `(TP, CONC, KV_OFFLOADING)` point.

## Cleanup

```bash
docker stop agentx-qwen35-fp4-mi355x
docker rm agentx-qwen35-fp4-mi355x
```

If you set `CONTAINER_NAME`, stop that name instead.
