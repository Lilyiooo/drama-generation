
# 剧本评测器

本目录收录 `Tencent-drama/drama_evaluator` 的评测代码、评分提示词和方法文档。完整方法实验使用本评测体系，以本地 `Qwen3.8-27B` 作为固定评测模型；剧本生成与状态抽取使用另一个模型，二者不在这里混用。

## 命令行使用

在仓库根目录运行；先安装 `drama_evaluator/requirements.txt`，并启动兼容 OpenAI Chat Completions 的评测服务。

```bash
export DRAMA_EVAL_BASE_URL=http://127.0.0.1:8001/v1
export DRAMA_EVAL_API_KEY=EMPTY
python drama_evaluator/evaluate_multi_agent.py /path/to/script.txt \
  --output-dir /path/to/evaluation_output \
  --model Qwen3.8-27B \
  --context-mode direct \
  --direct-char-limit 60000 \
  --chunk-chars 24000 \
  --max-output-tokens 16000
```

`evaluate_multi_agent.py` 是多评审、审核、按需仲裁的入口；评分细则保存在 `prompts/`，详细设计见 `PIPELINE.md`。真实实验还会由 `local_pipeline/drama_recover.py` 管理导出、模型配置和异常恢复。

`badcase_scripts/` 中的外部完整剧本没有上传；它们并非运行评测器所必需。运行结果、Web 上传内容、`.env` 和任何 API 密钥也未上传。`run_badcase_*.sh` 是历史测试脚本，若要复现对应测试，需自行提供有使用权限的测试剧本。
