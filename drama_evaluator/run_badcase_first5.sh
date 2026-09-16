#!/usr/bin/env bash
set -Eeuo pipefail

# 顺序评测 badcase_scripts 中编号 1～5 的剧本。
#
# 用法：
#   ./run_badcase_first5.sh
#   ./run_badcase_first5.sh --dry-run
#   EVAL_MODEL=gpt-5.6-sol REVIEW_WORKERS=1 ./run_badcase_first5.sh
#   ./run_badcase_first5.sh --restart
#
# 传给本脚本的参数会原样传递给 evaluate_multi_agent.py。

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_DIR/badcase_scripts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output/badcase_first5}"
EVAL_MODEL="${EVAL_MODEL:-gpt-5.6-sol}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REVIEW_WORKERS="${REVIEW_WORKERS:-3}"
AUDIT_WORKERS="${AUDIT_WORKERS:-3}"
ARBITRATION_WORKERS="${ARBITRATION_WORKERS:-3}"
CONTEXT_MODE="${CONTEXT_MODE:-auto}"

if [[ ! -d "$SCRIPT_DIR" ]]; then
  echo "错误：剧本目录不存在：$SCRIPT_DIR" >&2
  exit 1
fi

mkdir -p -- "$OUTPUT_ROOT"
cd -- "$PROJECT_DIR"

safe_model="${EVAL_MODEL//\//_}"
safe_model="${safe_model//\\/_}"
safe_model="${safe_model//:/_}"

for number in 1 2 3 4 5; do
  shopt -s nullglob
  matches=("$SCRIPT_DIR/$number "*.txt)
  shopt -u nullglob

  if (( ${#matches[@]} != 1 )); then
    echo "错误：编号 $number 应恰好匹配一个 TXT，实际匹配 ${#matches[@]} 个。" >&2
    exit 1
  fi

  script_path="${matches[0]}"
  filename="$(basename -- "$script_path" .txt)"
  title="${filename#"$number "}"
  title="${title%%全集剧本_*}"
  output_dir="$OUTPUT_ROOT/$(printf '%02d' "$number")_${title}_${safe_model}"

  echo "============================================================"
  echo "[$number/5] 开始评测：$script_path"
  echo "模型：$EVAL_MODEL"
  echo "输出：$output_dir"
  echo "============================================================"

  "$PYTHON_BIN" evaluate_multi_agent.py "$script_path" \
    --model "$EVAL_MODEL" \
    --output-dir "$output_dir" \
    --review-workers "$REVIEW_WORKERS" \
    --audit-workers "$AUDIT_WORKERS" \
    --arbitration-workers "$ARBITRATION_WORKERS" \
    --context-mode "$CONTEXT_MODE" \
    "$@"

  echo "[$number/5] 完成：$title"
done

echo "============================================================"
echo "前5部剧本已全部评测完成。"
echo "结果目录：$OUTPUT_ROOT"
echo "============================================================"
