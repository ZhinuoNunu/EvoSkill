#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export ANTHROPIC_AUTH_TOKEN="在此填写你的 token"
export ENABLE_TOOL_SEARCH=false
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"

uv run python scripts/run_loop.py \
  --sdk claude \
  --model claude-sonnet-4-6 \
  --mode skill_only \
  --max_iterations 3 \
  --concurrency 1 \
  --dataset ".dataset/officeqa_full.csv" \
  --num_samples 1 \
  "$@"
