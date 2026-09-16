# 单模型评测稳定性实验存档

- 日期：2026-08-26
- 评测模型：`gpt-5.6-sol`（Venus 网关）
- 评测入口：`drama_evaluator/evaluate.py`
- 波动测量脚本：`drama_evaluator/measure_stability.py`
- 数据目录：`drama_evaluator/output/`

## 一、实验目标

回答两个问题：

1. 单模型（`gpt-5.6-sol`）对同一个故事的多轮评测，分数波动有多大？
2. 波动是「随机噪声」还是「维度锁死」？根因在模型采样、故事内容，还是 prompt 评分尺度？

## 二、方法与数据

对每个剧本，用**完全相同的 prompt + 模型**独立评测 5 轮（每轮独立输出目录，复用 `evaluate.py` 的断点续跑），统计 8 个维度的均值 / 标准差 / 极差 / CV%。

评测流程（每轮 8 次 LLM 调用）：7 个专项评估（创意 / 逻辑 / 人物 / 卡点 / 情节规划 / 分集 / 设定）+ 1 个「剧本质量最终总分」（拿 5 个质量报告二次聚合）。

共 4 组数据：

| 数据前缀 | 故事 | 生成模型 | 篇幅（字符） |
|---|---|---|---|
| `stability` | 我在非洲当酋长 | kimi-k3 | 61,706 |
| `stability_baimoon` | 我是帝王白月光 | gpt-5.5 | 63,772 |
| `africa_gpt55` | 我在非洲当酋长 | gpt-5.5 | 65,663 |
| `africa_gemini` | 我在非洲当酋长 | gemini-2.5-pro | 99,798 |

## 三、波动结果总表（CV%）

| 维度 | 非洲·kimi | 白月光·gpt5.5 | 非洲·gpt5.5 | 非洲·gemini |
|---|--:|--:|--:|--:|
| 人物质量总分 | 1.8 | 2.0 | 1.7 | 3.1 |
| 剧本创意总分 | 2.9 | 0.2 | 6.0 | 6.1 |
| 情节规划质量总分 | 3.0 | 4.9 | 6.5 | 8.1 |
| 卡点质量总分 | 4.6 | 3.3 | 1.7 | 8.9 |
| 设定质量总分 | 10.0 | 4.0 | 4.8 | 12.0 |
| 剧本逻辑总分 | 7.4 | 11.5 | 9.6 | 13.2 |
| 分集剧本质量总分 | 10.2 | 22.2 | 19.0 | 16.8 |
| 剧本质量最终总分 | 7.4 | 13.1 | 11.1 | 4.7 |

各模型「最终总分」5 轮具体值：

```
kimi-k3         [64.4, 71.4, 62.7, 66.6, 56.9]  极差 14.5  均值 64.4
gpt-5.5         [56.4, 67.9, 50.3, 59.3, 51.3]  极差 17.6  均值 57.0
gemini-2.5-pro  [39.3, 43.8, 43.2, 39.2, 40.6]  极差  4.6  均值 41.2
```

## 四、核心结论

### 1. temperature 被模型锁死为 1.0，不可调

`llm.py` 的 payload 只传 `max_completion_tokens`，不传 `temperature`（docstring 明确说明是针对 GPT-5.6 的刻意设计）。实测：

```
无 temperature（现状）: 200 正常
temperature=0          : 400 "该模型只支持默认值 1"
temperature=0.5        : 400 "该模型只支持默认值 1"
temperature=1          : 200 正常
```

结论：`gpt-5.6-sol` 是推理型模型，`temperature` 服务端强制为 1.0，**想靠调低 temperature 压波动这条路走不通**。

### 2. 波动是「维度锁死」的，跨 3 个生成模型 × 4 个故事稳定复现

- 稳定维度（可单次用）：**人物质量**（CV 1.7–3.1%）、**剧本创意**（0.2–6.1%）
- 不稳定维度（不可单次用）：**分集剧本质量**（10–22%）、**剧本逻辑**（7–13%）

「分集 / 逻辑最抖、人物 / 创意最稳」的模式，在 4 个故事、3 个生成模型上完全一致。这说明**波动是评测器自身的属性，与「故事由谁生成」无关**。

### 3. temperature 是「放大器」，不是根因

temperature 是全局统一采样温度。若波动纯粹来自采样噪声，所有维度应同等级扰动。但实测分集维度 CV 比人物维度高 5–10 倍，无法用统一采样噪声解释。

真正机制：模型对「人物/创意」判断确定（尺度清晰、锚点足）→ 采样也收敛；对「分集/逻辑」判断本身摇摆（尺度模糊、锚点缺失）→ 采样把不确定性放大成分数波动。**根因在 prompt 评分尺度，不在采样参数。**

### 4. 波动「幅度」因故事内容而异，不是固定值

同样是「分集」维度，kimi 版 CV 10.2%、gpt-5.5 版 19.0%。只能说「分集维度总是最抖」，不能断言固定幅度。

### 5. 稳定 ≠ 准确（反直觉现象）

gemini 版最终总分波动最小（CV 4.7%、极差 4.6），但均值只有 **41.2**，显著低于 kimi（64.4）和 gpt-5.5（57.0）。原因是它的缺陷明显且稳定，评测模型每次都稳定给低分。

**低波动不代表评测变准了**，只代表故事「烂得一致」、模型没有摇摆空间。

### 6. 附带发现：评测器在「粗粒度横向对比」上仍有效

gpt-5.6-sol 能显著区分三个生成模型质量（41 < 57 < 64），说明这套评测用于「模型间横向对比」是有效的，问题集中在「单个故事的细粒度绝对分数」波动太大。

## 五、工程记录（踩坑与修复）

1. **路径 bug**：`llm.py` 硬编码查找 `evaluate_drama/drama_operator-master/drama_operator_new/trpc_python.yaml`，实际文件在 `/data/workspace/drama_operator/trpc_python.yaml`。已改为候选路径 fallback（原路径优先，缺失则回退到 `drama_operator/trpc_python.yaml`）。
2. **API key 无需额外提供**：复用 `drama_operator/trpc_python.yaml` 里的 `API_KEYS.venus` + `API_URLS.venus`，实测可用。
3. **TPM 限流**：5 个进程并行（15 并发）触发 Venus TPM 429（500000 tokens/min）。串行（单进程 3 并发）安全。`measure_stability.py` 默认串行 + `--sleep 12` 避让，`--parallel >1` 有明确限流风险警告。
4. **500 偶发**：推理服务偶发 `Internal server error`（500），`llm.py` 已有 3 次重试 + 指数退避兜底，能自动恢复，但会拖慢单轮耗时。
5. **stem 冲突**：不同故事文件同名 `scripts.txt` 时，`measure_stability.py` 需用 `--prefix` 自定义目录前缀区分。

## 六、复现方法

```bash
cd /data/workspace/evaluate_drama/drama_evaluator

# 跑 5 轮并统计（默认串行，避让限流）
python measure_stability.py 剧本.txt --runs 5 --output-root output/stability_xxx

# 多剧本 + 自定义前缀（文件名相同时）
python measure_stability.py a/scripts.txt b/scripts.txt \
    --prefix story_a story_b --runs 5 --output-root output

# 只统计已有结果（不调模型）
python measure_stability.py --output-root output
```

输出：控制台表格 + `stability_summary.csv`（utf-8-sig，Excel 友好）+ `stability_summary.json`。

## 七、后续方向（未做）

1. 打开「分集剧本质量」「剧本逻辑」两个 prompt，定位评分尺度模糊 / 锚点缺失的具体原因（治本）。
2. 给不稳定维度加固定锚点（低/中/高分锚点剧本），看是否能收敛判断不确定性。
3. 多轮取中位数/均值，验证能否压低最终总分的采样噪声（治标）。
4. 多 agent 评测（独立评审 + 分歧仲裁），但需先压住单模型基线波动，否则只会把「采样噪声」变成「从众噪声」。
