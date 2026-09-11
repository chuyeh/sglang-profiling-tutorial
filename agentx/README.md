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

## Torch profiling an AgentX point

Use a **separate profiling run** from the benchmark result you intend to report. Torch profiling, especially with stacks and input shapes, perturbs latency and throughput.

The convenience wrapper accepts the same five positional arguments as the colleague script:

```bash
# HBM-only TP4 point on GPUs 4-7.
SGLANG_ROOT=/path/to/sglang \
  ./run_profile_point.sh my-change 4 32 4,5,6,7 none

# TP4 HiCache point. The DRAM budget is required for metadata.
SGLANG_ROOT=/path/to/sglang DRAM_GB=1199 \
  ./run_profile_point.sh my-change 4 56 4,5,6,7 dram
```

Profiling defaults:

| Variable | Default | Meaning |
|---|---:|---|
| `DURATION` | `1500` | AIPerf measured duration |
| `PROFILE_DELAY_S` (`WARM`) | `840` | Delay **after AIPerf’s measured phase starts** |
| `PROFILE_WINDOW_S` (`WINDOW`) | `12` | Torch-profiler capture duration |
| `PROFILE_MIN_RUNNING_REQUESTS` | `1` | Do not start during an AgentX idle gap |
| `PROFILE_ACTIVITIES` | `CPU,GPU` | SGLang profiler activities (`GPU` is correct for ROCm PyTorch) |
| `PROFILE_WITH_STACK` | `1` | Collect operator source stacks |
| `PROFILE_RECORD_SHAPES` | `1` | Collect operator input shapes |
| `STOP_CONTAINER_AFTER` | `0` | Leave the server available for inspection |

The adapted controller does not rely only on time since server health. It:

1. follows the new AIPerf log until `Phase profiling (profiling) started`;
2. waits `PROFILE_DELAY_S`;
3. polls `sglang:num_running_reqs` so an inter-turn idle gap is not captured;
4. calls `/start_profile`, captures for `PROFILE_WINDOW_S`, calls `/stop_profile`;
5. fails if health, phase detection, endpoint calls, or trace export fail.

To keep launch and replay manually split, pass the same environment to both:

```bash
export TP=4 CONC=56 GPUS=4,5,6,7
export KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache TOTAL_CPU_DRAM_GB=1199
export RESULT_DIR=$PWD/../campaigns/my-profile-tp4-c56
export ENABLE_TORCH_PROFILER=1 PROFILE_DELAY_S=300 PROFILE_WINDOW_S=12

./qwen3.5_fp4_mi355x_launch_server.sh
./qwen3.5_fp4_mi355x_run_client.sh
```

Use a fresh `RESULT_DIR` for every capture. A short multi-rank trace can still be large; reduce `PROFILE_WINDOW_S`, or disable `PROFILE_WITH_STACK` / `PROFILE_RECORD_SHAPES`, when only kernel timing is needed.

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
