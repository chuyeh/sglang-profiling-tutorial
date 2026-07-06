#!/usr/bin/env bash
set -euo pipefail

# ===================== User-adjustable params =====================
IMAGE="lmsysorg/sglang:v0.5.12-rocm720-mi35x"
CONTAINER_NAME="wesley-sglang-profiling-kickstart"

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
  /bin/bash
