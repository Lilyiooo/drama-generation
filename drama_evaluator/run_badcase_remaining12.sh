#!/usr/bin/env bash
set -Eeuo pipefail

# 顺序评测 badcase_scripts 中编号 6～17 的剩余 12 部剧本。
#
# 用法：
#   ./run_badcase_remaining12.sh
#   ./run_badcase_remaining12.sh --dry-run
#   REVIEW_WORKERS=1 AUDIT_WORKERS=1 ARBITRATION_WORKERS=1 ./run_badcase_remaining12.sh
#   ./run_badcase_remaining12.sh --restart
#
# 传给本脚本的参数会原样传递给 evaluate_multi_agent.py。

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$PROJECT_DIR/badcase_scripts}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/output/badcase_remaining12}"
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

for number in {6..17}; do
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
  progress=$((number - 5))

  echo "============================================================"
  echo "[$progress/12] 开始评测：$script_path"
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

  echo "[$progress/12] 完成：$title"
done

echo "============================================================"
echo "剩余 12 部剧本已全部评测完成。"
echo "结果目录：$OUTPUT_ROOT"
echo "============================================================"
