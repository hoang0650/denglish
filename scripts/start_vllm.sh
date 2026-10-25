#!/usr/bin/env bash
# Serve Qwen3.6-35B-A3B (AWQ, Apache-2.0) with vLLM on the pod for action "llm".
# vLLM pins its own torch, so it lives in its own venv on the volume:
#   python -m venv /runpod-volume/venvs/vllm && /runpod-volume/venvs/vllm/bin/pip install -U vllm
# 48 GB GPU: weights ~26 GB. Give vLLM ~60% and leave the rest for light jobs
# (whisper, embed, rerank, birefnet). Do not run video/image jobs on the same GPU
# while vLLM is up; use a second pod for those.
set -euo pipefail
VOLUME="${AIMX_VOLUME:-/runpod-volume}"
VENV="${AIMX_VLLM_VENV:-$VOLUME/venvs/vllm}"
MODEL_DIR="${AIMX_VLLM_MODEL_DIR:-$VOLUME/qwen3.6-35b-a3b-awq}"
ARGS=(
  serve "$MODEL_DIR"
  --served-model-name "${AIMX_VLLM_MODEL:-qwen3.6-35b-a3b}"
  --host "${AIMX_VLLM_HOST:-127.0.0.1}"
  --port "${AIMX_VLLM_PORT:-8001}"
  --max-model-len "${AIMX_VLLM_MAX_LEN:-32768}"
  --gpu-memory-utilization "${AIMX_VLLM_GPU_UTIL:-0.60}"
  --enable-prefix-caching
  --reasoning-parser qwen3
)
if [[ -n "${AIMX_VLLM_API_KEY:-}" ]]; then
  ARGS+=(--api-key "$AIMX_VLLM_API_KEY")
fi
exec "$VENV/bin/vllm" "${ARGS[@]}"
