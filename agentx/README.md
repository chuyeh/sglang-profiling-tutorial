# Local AgentX (Qwen3.5 MXFP4 / MI355X / SGLang MTP)

Split InferenceX’s combined AgentX recipe into a **server launch** script and an **AIPerf client** script so you can keep SGLang up, rerun the benchmark, and stop the container yourself.

The manual split workflow defaults to a short local run. The one-click sweep below pins the official 3600-second recipe and image, but it is still a private reproduction rather than an InferenceX CI submission.

Upstream recipe: [qwen3.5_fp4_mi355x_sglang_mtp.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/main/benchmarks/single_node/agentic/qwen3.5_fp4_mi355x_sglang_mtp.sh).

## Prerequisites

- Docker with ROCm devices (`/dev/kfd`, `/dev/dri`)
- Image `rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260906` for manual local runs; the sweep pulls its pinned official image
- Weights: `/data2/amd/Qwen3.5-397B-A17B-MXFP4`
- Traces: `/data2/huggingface/dataset/cc-traces-weka-062126-256k/traces.jsonl`
- InferenceX checkout: `/home/chuyeh/workspace/InferenceX`
- AIPerf tree at pin `754356e9`: `/data2/zijchen/InferenceX_main_20260908/utils/aiperf`

Do **not** run the upstream combined recipe. Its EXIT trap kills the server after the client.

## One-click remote sweep

On an isolated 8-GPU MI355X server, start with:

```bash
cd /path/to/sglang-profiling-tutorial/agentx
./run_official_qwen35_agentx_sweep.sh
```

No per-run environment exports are needed. The runner:

- discovers the model, trace dataset, InferenceX, and AIPerf at common `/data2` locations;
- bootstraps the pinned InferenceX/AIPerf source when it is absent;
- pulls `lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260829`;
- verifies Docker, eight ROCm devices, enough host DRAM and result-disk space, an idle `/dev/kfd`, and a free port;
- runs each point sequentially in a fresh container with `DURATION=3600`, ten warmup requests per lane, EP1, MTP, and the official TP2/TP4 DRAM budgets;
- turns the torch profiler off, validates each aggregate, checkpoints it, and cleans up the container;
- stops starting work before the 23-hour deadline and writes `summary.csv` plus `summary.json`.

Inspect everything without pulling an image or starting a benchmark:

```bash
./run_official_qwen35_agentx_sweep.sh --plan
```

Validate Docker, GPUs, memory, disk, ports, image, and campaign state without starting a point:

```bash
./run_official_qwen35_agentx_sweep.sh --mode full --preflight
```

The default `24h` mode selects all ten TP2 points and six representative TP4 points. Based on the measured average of about 78 minutes per point, those 16 points take roughly 21 hours. It is a reduced official-compatible matrix, not the complete published curve.

The complete official matrix has 24 points and takes about 31 hours sequentially. InferenceX requests the node with `--exclusive`, so running TP2 and TP4 jobs side by side would introduce shared-node interference and would not be a like-for-like reproduction. Run the full matrix across two bookings instead:

```bash
./run_official_qwen35_agentx_sweep.sh --mode full
```

When the deadline guard pauses it, release the machine. Run the exact same command in the next booking; validated points are skipped automatically. You can also run the default mode first and later switch to `--mode full` because both modes share checkpoints.

If the local machine is continuously available for more than 31 hours, run the full sweep in one session:

```bash
./run_official_qwen35_agentx_sweep.sh --mode full --hours 48
```

Start only when every GPU is idle. The preflight rejects VRAM left on any device because InferenceX uses an exclusive node for these measurements.

The clock starts when the command starts. Launch it near the beginning of the reservation; if only 12 hours remain, pass `--hours 12` and it will stop scheduling points earlier.

Weights and traces must be staged before the timed booking; automatically downloading a 397B checkpoint would consume an unpredictable fraction of it. The runner verifies model revision `edf0958bc373` by its config/index checksums and trace revision `8fecd2fc5669` by its data checksum, so it fails early instead of benchmarking the wrong snapshot. If the paths are unusual, save them once instead of exporting variables in every shell:

```bash
cp machine.conf.example machine.conf
$EDITOR machine.conf
./run_official_qwen35_agentx_sweep.sh --plan
```

The runner prefers a writable data-disk campaign such as `/data2/models/agentx-runs/$USER/agentx-qwen35-official-mi355x-v0518/` and falls back to `../campaigns/`. The exact path is printed by `--plan`. Important files are:

- `manifest.tsv`: all official points and the points selected by the latest mode;
- `status.tsv` / `sweep.log`: scheduler history and complete console log;
- `points/<point>/agentx_result.json`: the clean aggregate for one point;
- `summary.csv` / `summary.json`: dashboard-oriented throughput, interactivity, TTFT, cache, and list-price token-per-dollar fields.

Power collection is disabled in this Docker workflow. The container can see all eight host devices while a TP2/TP4 aggregate expects two/four, which makes InferenceX correctly reject that telemetry as a GPU-count mismatch. This does not affect throughput, latency, or list-price token-per-dollar metrics.

If a clean point is problematic, rerun that exact checkpoint with profiling rather than profiling the sweep:

```bash
CAMPAIGN=/path/printed/by/plan
POINT=tp4-c64-hicache
source "$CAMPAIGN/points/$POINT/point.env"

RESULT_DIR="$CAMPAIGN/profiles/$POINT" \
CONTAINER_NAME="agentx-profile-$POINT" \
STOP_CONTAINER_AFTER=1 \
  ./run_profile_point.sh "$POINT" "$TP" "$CONC" "$GPUS" "$KV_OFFLOADING"
```

Do not report the profiled rerun's aggregate as a benchmark result.

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
| `run_clean_point.sh` | Clean point lifecycle helper used by the sweep |
| `run_official_qwen35_agentx_sweep.sh` | Deadline-aware, resumable remote sweep |
| `summarize_sweep.py` | Builds sweep `summary.csv` and `summary.json` |
| `run_profile_point.sh` | One-point launch + replay + torch-profile convenience wrapper |

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

## Manual split workflow on another MI355X server

The scripts do not reserve GPUs 4-7; any physical GPU list is accepted. The number of IDs must equal `TP`. For example, use `GPUS=0,1,2,3` or `GPUS=4,5,6,7` for TP4. The official Qwen3.5 AgentX matrix uses TP2 and TP4; having eight GPUs available does not by itself make TP8 comparable to those published points.

The one-click sweep discovers these paths or reads `machine.conf`. Only the lower-level manual launch/client scripts require exports:

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
