# Prompt 索引

## 剧本生成请求

实际请求由两条消息组成：

1. system：`GENERATION_SYSTEM_PROMPT.txt`，其权威源码是 `../src/narrative_memory_feedback_v2_7_self_evolving/prompts.py` 中的 `SYSTEM_PROMPT`。
2. user：由同文件的 `build_generation_prompt()` 动态生成。

动态 user prompt 依次包含：

- 故事与人物
- 当前累积故事状态
- 当前分集目标与开放实现方向
- 集末连续性硬边界
- 当前开放叙事义务
- 可选 V4 创作技法和反模式门
- 动作化写作、自然叙事和成稿要求

经验卡注入文本不是写死在主模板里，而是由 `../src/narrative_memory_feedback_v2_7_self_evolving/retrieval_v4.py` 的 `v4_experience_block()` 生成，然后作为 `experience_text` 传入。

## 状态与义务抽取请求

同一 `prompts.py` 中：

- `build_extraction_prompt()`：首轮完整抽取
- `build_compact_extraction_prompt()`：校验失败后的紧凑重试

人物关系、人物因果痕迹等其他可选 prompt 分别在 `relationship_memory.py` 和 `character_trace.py`。本冻结条件主要启用生命周期、V4 卡和义务检索。

## 展开后的真实 Prompt

运行时 `config.json` 的 `runtime.store_prompts=true`，所以每一集发送给模型的完整 user prompt 都保存在：

```text
outputs/qwen36_full_best_v1/Sxx__Rxx/contexts.jsonl
```

按 `episode_id` 找对应记录并读取 `prompt` 字段即可。这里是最可靠的逐集审计来源，因为后续集 prompt 会包含该轨迹自己的动态状态和义务。
