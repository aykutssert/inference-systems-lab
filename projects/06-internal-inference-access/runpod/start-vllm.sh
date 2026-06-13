#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="${VLLM_RUNTIME_DIR:-/root/vllm-runtime}"
MODEL="${VLLM_MODEL:-Qwen/Qwen3-1.7B-GPTQ-Int8}"
PORT="${VLLM_PORT:-8000}"
VLLM_VERSION="${VLLM_VERSION:-0.22.1}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"

if curl --fail --silent "$HEALTH_URL" >/dev/null; then
  echo "vLLM is already healthy"
  exit 0
fi

if pgrep -f "[v]llm serve" >/dev/null; then
  echo "vLLM is already starting on port ${PORT}"
  exit 0
fi

mkdir -p "$RUNTIME_DIR"
if [[ ! -x "$RUNTIME_DIR/.venv/bin/vllm" ]]; then
  uv venv --python 3.12 "$RUNTIME_DIR/.venv"
  uv pip install \
    --python "$RUNTIME_DIR/.venv/bin/python" \
    "vllm==${VLLM_VERSION}" \
    --torch-backend=cu128
fi

export LD_LIBRARY_PATH="$RUNTIME_DIR/.venv/lib/python3.12/site-packages/nvidia/cu13/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

exec "$RUNTIME_DIR/.venv/bin/vllm" serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9
