#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export ANTHROPIC_AUTH_TOKEN="在此填写你的 token"
export ENABLE_TOOL_SEARCH=false
export ANTHROPIC_BASE_URL="https://api.kimi.com/coding/"

echo "========== Stage 1: run_loop =========="
uv run python scripts/run_loop.py \
  --sdk claude \
  --model claude-sonnet-4-6 \
  --mode skill_only \
  --max_iterations 3 \
  --concurrency 1 \
  --train_dataset ".dataset/officeqa_easy_train.csv" \
  --val_dataset ".dataset/officeqa_easy_val.csv"

echo ""
echo "========== Stage 2: run_eval =========="
uv run python scripts/run_eval.py \
  --sdk claude \
  --model claude-sonnet-4-6 \
  --dataset_path ".dataset/officeqa_easy_test.csv" \
  --difficulty all \
  --max_concurrent 1 \
  --output "results/eval_results.pkl"
