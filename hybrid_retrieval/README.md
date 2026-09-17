# Hybrid 状态检索（60 集大纲实验）

本目录收录一项与根目录默认 V4 卡 + 义务记忆生成器隔离的实验：`S1_lifecycle` 纯状态周期记忆下，比较完整状态与 hybrid 状态检索。

## 实验协议

- 输入：[`data/episode_outlines.json`](data/episode_outlines.json)，一条连续的 60 集古装故事大纲。
- 模型：Qwen3.6-27B 负责生成和状态抽取，关闭 thinking；固定 Qwen3.8-27B 负责整剧评测。
- 分组：`full` 与 `hybrid`，均为 `R01`--`R03` 三次运行；同一 `run + episode` 跨臂使用同一确定性随机种子。
- `full`：把完整当前状态放进生成 prompt。
- `hybrid`：以当前集大纲为 query，在最多 6000 个 JSON 字符内，联合词项相关性、时间邻近性与状态字段权重选择状态条目。

两臂只保留每集九字段状态更新与生命周期清理。义务记忆/检索、经验/策略卡、MemoryPool、关系记忆、人物痕迹、写回和前集剧本文本均关闭。故这里检索的是**状态条目**，不是义务、经验卡或前文原文。

## 实现与结果

- [`state_selector.py`](state_selector.py) 是实际使用的确定性状态选择器；本实验仅调用 `full` 和 `hybrid` 两个臂。
- [`results/summary.json`](results/summary.json) 是 6 条完整 60 集剧本的固定评测汇总。
- [`results/retrieval_context_stats.json`](results/retrieval_context_stats.json) 记录状态上下文成本：hybrid 在其自身轨迹上平均每集注入 5,258 个字符，相比完整状态减少 59.82%。

| 方法 | 逻辑 | 质量 | 创造性 |
|---|---:|---:|---:|
| `full` | 66.740 | 81.123 | **59.367** |
| `hybrid` | **67.527** | **82.660** | 58.773 |
| `hybrid - full` | +0.787 | +1.537 | -0.593 |

仅有三次配对运行，这些是描述性结果，不应表述为统计显著性结论。

## 复现接入

根目录 `src/narrative_memory_feedback_v2_7_self_evolving/` 提供生成引擎。对每条独立轨迹，应在生成前将完整状态与当前集 plan 传给：

```python
from hybrid_retrieval.state_selector import select_state

state_view, audit = select_state(
    full_state,
    initial_state,
    prior_state_updates,
    current_episode_plan,
    arm='hybrid',
    budget=6000,
)
```

`full` 臂直接传入 `full_state`；`hybrid` 臂把 `state_view` 传给同一生成 prompt。为公平比较，两臂必须共享同一 `run + episode` seed、计划、大纲、抽取与生命周期实现。生成的 `audit` 应逐集保存，记录选择条目、完整/选中字符数和 query，便于计算上下文成本并验证预算。

真实运行输出、API 调用记录、剧本文本和日志不随本目录上传；模型权重也不包含在仓库中。
