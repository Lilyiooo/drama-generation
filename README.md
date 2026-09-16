# 最佳已验证完整剧本生成器（2026-09-16 快照）

本目录整理的是当前已经完成 6 个故事 × 40 集 × 3 次运行，并经过固定 Qwen3.8 drama_evaluator 评测的最佳完整生成方法：

- 方法条件：`S1_lifecycle_v4_obligation_retrieval`
- 生成与状态抽取：`Qwen3.6-27B`
- 思考模式：关闭
- 每集目标长度：1200–2200 中文字符
- 生成参数：temperature=0.7、top_p=1.0、max_output_tokens=3600
- 抽取参数：temperature=0、max_output_tokens=4096
- 上下文：故事设定、当前累积状态、当前分集大纲、开放义务、生命周期、V4 卡选择/反模式门
- 输出：40 集剧本、逐集 prompt、状态抽取、生命周期、义务、卡检索和 API 调用记录

历史完整实验共 18 条方法轨迹，对完整前文裸模型的 Qwen3.8 固定评测均值为：

| 方法 | 逻辑 | 质量 | 创意 |
|---|---:|---:|---:|
| 本方法 | 70.22 | 79.81 | 59.32 |
| 裸模型（提供全部前集原文） | 58.02 | 61.97 | 44.58 |

这些是同一批 6 个故事、3 个 run 的历史实验均值，不是对任意新故事的性能保证。

## 仓库内的历史输出

`outputs/historical_40ep/` 同时收录上述已完成实验的当前方法（`method`）和裸模型（`baseline`），每组 S01–S06 × R01–R03 × 40 集。`baseline` 生成时提供了此前**全部集的剧本原文**，不是只有大纲。

每条轨迹均有逐集 `generations.jsonl`、实际展开的 `contexts.jsonl`、方便阅读的 `script.txt`、固定 Qwen3.8 的 `scores.json`、`jobs.jsonl` 与 `execution_protocol.json`。顶层 `manifest.json` 记录原路径、文件 SHA-256、集数和两组均分。生成阶段的其他中间日志、完整评测工作目录和模型权重**没有上传**；需要复核评分细节时请使用原始 `runs/qwen36_fullmethod_vs_bare_40ep/`。

这些历史输出是只读研究材料；新的运行结果仍写入并忽略 `outputs/qwen36_full_best_v1/`，不会把不断增长的剧本或日志意外推送到 GitHub。

## 为什么这是默认版本

这里的“最佳”指当前有完整 40 集成品和固定评测证据的版本。`qwen36_gate_branch_v2` 的新门控提示词只完成了短分支生成，尚未完成评分，而且两个分支使用过引用约束补跑，因此没有合入这个默认快照。等配对评测证明它稳定更好后，应作为新版本单独冻结，不能覆盖本目录。

## 目录结构

```text
best_full_script_generator_20260916/
├── README.md
├── VERSION.json
├── run_full.sh
├── run_full.py
├── extraction_retry.py
├── verify_snapshot.py
├── prompts/
│   ├── GENERATION_SYSTEM_PROMPT.txt
│   └── PROMPT_INDEX.md
├── provenance/
│   ├── batch_experiment.py
│   └── historical_experiment.json
└── src/
    └── narrative_memory_feedback_v2_7_self_evolving/
        ├── config.json
        ├── prompts.py
        ├── run_generation.py
        ├── retrieval_v4.py
        └── data/
            ├── stories.json
            ├── episode_plans.json
            ├── memory_pools_v4.json
            └── narrative_move_taxonomy.json
```

`src` 内是完整原始 Python 包，不依赖 `Tencent-drama` 其他生成源码。Conda 环境不复制，仍使用已有的 `qwen36-vllm`。

## 先验证快照

```bash
cd /inspire/hdd/global_user/wangqiqi-CZXS25210124/Tencent-drama/best_full_script_generator_20260916
bash run_full.sh verify
```

应输出：

```text
snapshot_ok=true
historical_source_fingerprint=3717ec06b0d2d613d33b1da66fba63b9998a866422cec1baaac9d5ce0ab609df
```

## 生成完整剧本

先在 GPU 机器上启动 Qwen3.6 vLLM 服务，然后在另一个终端运行：

```bash
cd /inspire/hdd/global_user/wangqiqi-CZXS25210124/Tencent-drama/best_full_script_generator_20260916

bash run_full.sh generate \
  --stories S01,S02,S03,S04,S05,S06 \
  --runs R01,R02,R03 \
  --workers 4 \
  --execute-api \
  2>&1 | tee -a outputs/generation.log
```

默认输出到：

```text
best_full_script_generator_20260916/outputs/qwen36_full_best_v1/
```

每个 worker 独占一条 40 集轨迹；同一目录不要同时启动两个调度器。失败后重跑完全相同的命令，会从已经验证的 JSONL 断点继续，不覆盖成功集。

只生成一个故事的一次运行：

```bash
bash run_full.sh generate --stories S01 --runs R01 --workers 1 --execute-api
```

查看进度：

```bash
bash run_full.sh status
```

导出便于阅读的整剧 TXT：

```bash
bash run_full.sh export
```

TXT 位于每条轨迹的 `exported_scripts/`，逐集原始数据位于 `generations.jsonl`。

## Prompt 在哪里

- 生成 system prompt：`prompts/GENERATION_SYSTEM_PROMPT.txt`
- 完整动态生成模板与抽取 prompt：`src/narrative_memory_feedback_v2_7_self_evolving/prompts.py`
- V4 Strategy/Anti-pattern/Repair 注入文本：`src/narrative_memory_feedback_v2_7_self_evolving/retrieval_v4.py` 的 `v4_experience_block`
- Prompt 的组装调用：`src/narrative_memory_feedback_v2_7_self_evolving/run_generation.py`
- 每次真实运行展开后的逐集完整 user prompt：对应轨迹的 `contexts.jsonl` 中 `prompt` 字段

不要只复制静态 user prompt：从第二集开始，状态、开放义务、卡片和反模式门都会根据上一集实际剧本动态变化。

## 大纲和经验卡

- 总设定：`src/narrative_memory_feedback_v2_7_self_evolving/data/stories.json`
- 6×40 集分集大纲：`src/narrative_memory_feedback_v2_7_self_evolving/data/episode_plans.json`
- 当前 40 张 V4 卡：`src/narrative_memory_feedback_v2_7_self_evolving/data/memory_pools_v4.json`
- 叙事动作分类：`src/narrative_memory_feedback_v2_7_self_evolving/data/narrative_move_taxonomy.json`

修改任何上述文件后都不再是这个冻结版本。应复制整个目录、更新版本号和验证记录，而不是直接覆盖。

## 抽取重试说明

`extraction_retry.py` 只在普通抽取输出未通过校验后的 retry 请求上增加 JSON Schema 和合法 move_id 枚举。它不改变生成 prompt、剧本文本、卡选择或状态生命周期规则，也不会自动把无效内容当作有效。这个机制来自完整实验的补跑流程，用于减少格式错误造成的轨迹中断；调用记录保存在 `recovery_extraction_responses.jsonl`。
