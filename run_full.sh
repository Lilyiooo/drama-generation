#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ENV="${VLLM_ENV:-/inspire/qb-ilm/project/exploration-topic/wangqiqi-CZXS25210124/anaconda3/envs/qwen36-vllm}"

export PYTHONDONTWRITEBYTECODE=1
export NMF_GENERATION_BASE_URL="${NMF_GENERATION_BASE_URL:-http://127.0.0.1:8000/v1}"
export NMF_GENERATION_MODEL="${NMF_GENERATION_MODEL:-Qwen3.6-27B}"
export NMF_GENERATION_API_KEY="${NMF_GENERATION_API_KEY:-EMPTY}"
export NMF_EXTRACTION_BASE_URL="${NMF_EXTRACTION_BASE_URL:-$NMF_GENERATION_BASE_URL}"
export NMF_EXTRACTION_MODEL="${NMF_EXTRACTION_MODEL:-Qwen3.6-27B}"
export NMF_EXTRACTION_API_KEY="${NMF_EXTRACTION_API_KEY:-$NMF_GENERATION_API_KEY}"
export NMF_GENERATION_DISABLE_THINKING=true
export NMF_EXTRACTION_DISABLE_THINKING=true
export NMF_API_TIMEOUT_SECONDS="${NMF_API_TIMEOUT_SECONDS:-600}"

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  echo "Missing Python environment: $VLLM_ENV/bin/python" >&2
  exit 2
fi

exec "$VLLM_ENV/bin/python" "$BUNDLE_ROOT/run_full.py" "$@"
