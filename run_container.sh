#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
IMAGE="lmsysorg/sglang:v0.5.14-cu130"
CONTAINER_NAME="wesley-sglang-profiling-kickstart"

MODELS_DIR="/models"
PROFILING_DIR="$HOME/workspace/sglang-profiling-tutorial"
# Optional: mount your local sglang source to profile it instead of the
# version baked into the image (then run `pip install -e .` inside).
SGLANG_SRC="$HOME/workspace/sglang"

PORT=9001
# =================================================================

docker run -it --rm \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --ipc=host --shm-size 16g \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${MODELS_DIR}:${MODELS_DIR}" \
  -v "${PROFILING_DIR}:/workspace/profiling" \
  -v "${SGLANG_SRC}:/workspace/sglang" \
  -w /workspace/profiling \
  -p "${PORT}:${PORT}" \
  "${IMAGE}" \
  /bin/bash
