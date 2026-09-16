#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 我在非洲当酋长 —— 多模型 / 多次评测批跑脚本
# 评测模型默认 gpt-5.6-sol，可用环境变量 EVAL_MODEL 覆盖
# 输出目录规约：output/<剧本>_<生成模型>_<评测模型>[_Rxx]
# ============================================================

EVAL_MODEL="${EVAL_MODEL:-gpt-5.6-sol}"

PROJ="/data/workspace/drama_eval-demo/momomohe-multiagent"
OUT_ROOT="$PROJ/output"

SCRIPT_GEMINI="/data/workspace/十部/原创gpt5.5+gemini2.5/gemini-2.5-pro/我在非洲当酋长/scripts.txt"
SCRIPT_GPT55="/data/workspace/十部/原创gpt5.5+gemini2.5/gpt-5.5/我在非洲当酋长/scripts.txt"
SCRIPT_KIMI="/data/workspace/十部/kimi-k3/我在非洲当酋长.txt"

cd "$PROJ"

run_eval() {
  local script="$1"
  local out="$2"
  echo "============================================================"
  echo "[评测] 输入：$script"
  echo "[评测] 输出：$out"
  echo "[评测] 模型：$EVAL_MODEL"
  echo "============================================================"
  python evaluate_multi_agent.py "$script" \
    --model "$EVAL_MODEL" \
    --output-dir "$out" \
    --restart
}

# 1) gemini-2.5-pro 生成版，评 3 次，目录用 R01/R02/R03 分开
for run_no in 01 02 03; do
  run_eval "$SCRIPT_GEMINI" \
    "$OUT_ROOT/非洲酋长_gemini-2.5-pro_${EVAL_MODEL}_R${run_no}"
done

# 2) gpt-5.5 生成版，评 1 次
run_eval "$SCRIPT_GPT55" \
  "$OUT_ROOT/非洲酋长_gpt-5.5_${EVAL_MODEL}"

# 3) kimi-k3 生成版，评 1 次
run_eval "$SCRIPT_KIMI" \
  "$OUT_ROOT/非洲酋长_kimi-k3_${EVAL_MODEL}"

echo "============================================================"
echo "全部评测完成，结果目录：$OUT_ROOT"
echo "============================================================"
