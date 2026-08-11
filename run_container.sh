#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
# "lmsysorg/sglang-rocm:v0.5.12.post1-rocm720-mi35x-20260528" -> old image used in 06/05
# "lmsysorg/sglang:v0.5.14-rocm720-mi35x" -> baseline image used in 07/05
IMAGE="${IMAGE:-sglang:v0.5.14-rocm720-mi35x-pr24651}"
CONTAINER_NAME="wesley-sglang-profiling-qwen3.5-fp8"

MODELS_DIR="/raid/models"
PROFILING_DIR="$HOME/workspace/profiling"
# Optional: mount your local sglang source to profile it instead of the
# version baked into the image (then run `pip install -e .` inside).
SGLANG_SRC="$HOME/workspace/sglang"

PORT=9001
# =================================================================

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video \
  --ipc=host --shm-size 16g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${MODELS_DIR}:${MODELS_DIR}" \
  -v "${PROFILING_DIR}:/workspace/profiling" \
  -v "${SGLANG_SRC}:/workspace/sglang" \
  -w /workspace/profiling \
  -p "${PORT}:${PORT}" \
  "${IMAGE}" \
  bash -c 'pip install -e /workspace/profiling/torch-profiler-parser && exec /bin/bash'
