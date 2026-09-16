from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """你是专业中文连续剧编剧。先把约束转化为人物当下的欲望、阻力、行动和后果，再写戏。约束只用于幕后校验，绝不能被角色当作规则、创作说明或程序口号复述。对白必须来自人物关系和眼前情境。

人物性格只通过具体选择、动作和潜台词流露，绝不靠角色自述、旁白结论或复读身份标签来交代。用可观察的行为外化内心（show, don't tell）：让手势、站位、对物件的操作、停顿与回避去承载情绪与关系变化。"""


def block(value: Any) -> str:
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, indent=2)


def character_block(characters: list[dict[str, Any]]) -> str:
    return "\n".join(f"- {item['name']}：{item['role']}" for item in characters)


def character_trace_block(traces: dict[str, Any]) -> str:
    lines: list[str] = []
    emotional = traces.get("emotional_traces", [])
    agency = traces.get("agency_traces", [])
    if emotional:
        lines.append("情绪痕迹：")
        for item in emotional:
            lines.append(
                f"- {item['character']}：触发={item['trigger_event']}；理解={item['appraisal']}；"
                f"残留压力={item['residue']}；仍受威胁={item['unresolved_stake']}"
            )
    if agency:
        lines.append("主体性痕迹：")
        for item in agency:
            lines.append(
                f"- {item['character']}：选择={item['decision']}；动机来源={item['motivation_source']}；"
                f"放弃={item['rejected_alternative']}；已接受代价={item['accepted_cost']}；"
                f"待落地后果={item['pending_consequence']}"
            )
    if not lines:
        return "（无）"
    return "\n".join(lines)


def private_boundary_block(story: dict[str, Any], plan: dict[str, Any]) -> str:
    boundaries = {
        "character_continuity": [
            {"name": item["name"], "boundaries": item.get("constraints", [])}
            for item in story["characters"]
        ],
        "episode_end_boundaries": plan["hard_anchors"],
    }
    return block(boundaries)


def experience_block(cards: list[dict[str, Any]]) -> str:
    if not cards:
        return "没有额外技法参考；根据人物目标和当前阻力自主选择叙事机制。"
    return "\n\n".join(
        f"技法参考 {index}\n适用情境：{card['applicable_situation']}\n可选策略：{card['narrative_strategy']}\n可能效果：{card['expected_effect']}\n使用风险：{'；'.join(card['risks'])}"
        for index, card in enumerate(cards, start=1)
    )


def build_generation_prompt(*, story: dict[str, Any], plan: dict[str, Any], state: dict[str, Any], cards: list[dict[str, Any]], output_characters: dict[str, int], previous_script: str | None = None, obligations: list[dict[str, Any]] | None = None, relationship_memories: list[dict[str, Any]] | None = None, experience_text: str | None = None, character_traces: dict[str, Any] | None = None, character_framing: bool = False, character_trace_gate: str | None = None) -> str:
    prompt = f"""## 故事与人物
标题：{story['title']}\n类型：{story['genre']}\n世界观：{story['world_setting']}\n系列目标：{story['series_goal']}

以下人物条目是“处境与职责锚点”，用来约束这个人会做出什么选择，不是要角色复读的台词脚本，也不要把它当作性格判词直接写进人物独白：
{character_block(story['characters'])}

## 戏剧起点
以下是开场前已经发生或确认的事实。把它们当作人物生活的一部分，不要让角色为了复述背景而对话。\n{block(state)}

## 本集戏剧任务
集数：{plan['episode_id']}\n目标：{plan['episode_goal']}

可自由选择的实现方向：\n{block(plan['open_decisions'])}

## 幕后连续性边界
以下内容只供作者在完成剧本后检查集末状态。它们不是人物要说的话，也不是每条都要在场景中被提及。只要剧情结果不越过边界即可。\n{private_boundary_block(story, plan)}

## 当前开放叙事义务
以下义务是前序剧情已经建立、未来需要兑现的伏笔/谜题/承诺。它们是故事世界内的连续性债务，不要求本集全部解决；只有本集任务自然触及时才推进或兑现，不得生硬插入，也不得凭空改写。{"这些义务限定的是剧情连续性与兑现责任，不是替人物决定具体行动；人物仍要基于自己的目标、恐惧、关系与代价作出选择，可以拒绝、拖延或改写外部安排。" if character_framing else ""}\n{block(obligations or [])}

## 当前开放人物关系记忆
以下是前序剧情形成、仍未结清的人物关系状态记录；包含当事人、张力来源、当前立场、情绪残留、回应责任与关系边界。它们是人物之间真实的亏欠与张力，不要简化成可执行任务或计划，也不要求本集解决；只有本集自然触及时才推进。\n{block(relationship_memories or [])}

## 当前人物因果痕迹
{character_trace_block(character_traces) if character_traces else "（无）"}

## 人物层反模式门
{character_trace_gate or "（无）"}

## 可选创作技法
这些内容只影响写法，不是故事世界里的信息，也不要求逐项采用。\n{experience_text if experience_text is not None else experience_block(cards)}

## 写作核心
先把每条约束转成人物当下的欲望、阻力、行动与后果，再去写戏。约束只在幕后校验，绝不被角色当成规则、说明或口号复述。

## 用动作说话（show, don't tell）
- 人物内心、性格、关系变化，只通过可观察行为流露：手势、站位、对物件的操作、停顿、回避。绝不写“他是个…的人”这类判词，不让角色自报身份、职业或性格（如“我是个停职刑警，所以……”）。
- 每场都要有足可拍摄的肢体/环境动作，动作与场面指示占正文明显份额（建议不低于两成），避免“只有对话没有戏”。
- 职业边界与硬约束靠行为后果体现：某人真的先核实再落笔、另一人真的按流程等审批。严禁让角色用台词声明克制——反面典型：“我只看设备，碰不到案卷”“只限事实，不带推断”“你不是技术员，别碰设备”“按流程来，你别越界”。分寸用一句话或一个动作演出来，不要反复声明，更不要把冲突写成“你怎么能…/你凭什么…”式的程序性互相指责循环。

## 自然叙事
- 场景围绕当前人物的实际行动展开，后续发展与此前事件保持因果联系；冲突机制由当前状态决定，不预设固定模板。
- 标点服务语流：连续短句（少于 8 字）不超过全部分句的 40%，每分句以 10–20 字为宜，长短错落。好例：“巴图把清点册翻到最后一页，手指在褪色的草料收据上停了片刻，才慢慢抬起眼。” 差例（避免）：“巴图翻册。手指停。他抬头。”
- 分场可用“场景一/场景二”标题，但每集开头不要都用同一固定场景、地点与定场镜头起手；可从对话、动作、回忆或事件中途切入，时辰地点随剧情变化。

## 成稿要求
只输出当前一集可拍摄剧本，使用场景标题、具体动作、人物名和台词；长度为 {output_characters['min']} 至 {output_characters['max']} 个中文字符。结尾产生可继承的状态变化。
动作与场面指示要具体、可被执行，承载人物与情绪；对话服务于选择与关系，而非解释人物或交代背景。"""

    # feed_state=False（G_prev_only）时不喂抽象状态：移除“戏剧起点”空块，避免 prompt 出现空标题。
    if state is None:
        prompt = prompt.replace(
            "## 戏剧起点\n以下是开场前已经发生或确认的事实。把它们当作人物生活的一部分，不要让角色为了复述背景而对话。\n\n",
            "",
        )

    # 未启用人物层门控时移除空门控块，避免 prompt 出现无意义标题。
    if not character_trace_gate:
        prompt = prompt.replace("## 人物层反模式门\n（无）\n\n", "")

    # 未启用关系记忆时移除空关系记忆块（block([]) 渲染为 []）。
    if relationship_memories is None:
        prompt = prompt.replace(
            "## 当前开放人物关系记忆\n以下是前序剧情形成、仍未结清的人物关系状态记录；包含当事人、张力来源、当前立场、情绪残留、回应责任与关系边界。它们是人物之间真实的亏欠与张力，不要简化成可执行任务或计划，也不要求本集解决；只有本集自然触及时才推进。\n[]\n\n",
            "",
        )

    # feed_previous_script=True（G_prev_only）：把紧邻上一集原文作为“无抽象、原始上下文”对照，
    # 插到“本集戏剧任务”之前（仅当提供了上一集内容时）。
    if previous_script:
        prev_block = (
            "## 上一集剧本（紧邻上一集，仅供连续性参考）\n"
            "这是紧邻的上一集完整剧本，用于把握因果衔接与人物状态变化；"
            "请勿照搬其台词、句式或具体情节，本集须有独立叙事推进。\n"
            f"{previous_script}\n\n"
        )
        prompt = prompt.replace("## 本集戏剧任务\n", prev_block + "## 本集戏剧任务\n", 1)

    return prompt


def build_extraction_prompt(*, story: dict[str, Any], plan: dict[str, Any], previous_state: dict[str, Any], script: str, move_ids: list[str], open_obligations: list[dict[str, Any]] | None = None, trial_cards: list[dict[str, Any]] | None = None, experience_context: dict[str, Any] | None = None, exclude_relationship_debt: bool = False) -> str:
    prompt = f"""你是连续剧状态、Narrative Move 与程序性经验采用标注器。只根据剧本提取，不修复、不补写。只输出合法 JSON 对象，不要 Markdown。

故事：{story['title']}\n当前集：{plan['episode_id']}\n集末连续性边界：{block(plan['hard_anchors'])}\n上一状态：{block(previous_state)}\n当前开放叙事义务：{block(open_obligations or [])}\n本集经验形成上下文：{block(experience_context or {})}\n本集受控试用卡：{block(trial_cards or [])}\n允许的 move_id：{block(move_ids)}

剧本：\n{script}

输出格式：
{{
  "state_delta": {{"character_state": [], "relationship_state": [], "known_information": [], "unknown_information": [], "confirmed_facts": [], "unresolved_threads": [], "resources_and_evidence": [], "current_goals": [], "timeline": []}},
  "resolved_goals": [],
  "resolved_unknown": [],
  "retracted_facts": [],
  "new_obligations": [{{"type": "FORESHADOW|MYSTERY|PROMISE|PLAN|DEADLINE|RELATIONSHIP_DEBT", "description": "本集新建立的义务", "participants": [], "required_payoff": "未来怎样才算兑现", "deadline": null}}],
  "resolved_obligations": [],
  "primary_move_id": "允许的 move_id 之一",
  "secondary_move_ids": [],
  "move_evidence": "剧本中的可核查依据",
  "state_constraint_count": 0,
  "state_consistency_issues": [],
  "card_feedback": [{{"memory_id": "受控试用卡原始ID", "adopted": false, "evidence": "采用时逐字引用剧本证据，否则为空", "expected_effect_achieved": false, "state_supported": false, "anchor_supported": false, "causal_supported": false, "harm_flags": []}}],
  "writeback_candidate": {{"quality_score": 0.0, "confidence": 0.0, "card_type": "STRATEGY", "primary_move_id": "允许的 move_id 之一", "mechanism_family": "抽象英文大写机制名", "episode_functions": ["当前功能"], "applicable_phases": ["当前阶段"], "applicable_state_features": ["从本集经验形成上下文逐字选择2-3个标签"], "inapplicable_state_features": [], "applicable_situation": "不含故事专名的适用情境", "operator_steps": ["可执行步骤1", "可执行步骤2"], "expected_state_change": "可核查的预期状态变化", "observable_evidence": ["采用后应看到的证据"], "risks": [], "abort_conditions": [], "conflicts_with_invariants": []}}
}}

严格遵守以下压缩规则：
- state_delta 只记录本集相对上一状态真正新增或改变的内容，绝不能复制上一状态、复述人物履历或汇总此前各集。
- resolved_goals / resolved_unknown / retracted_facts 用于「清除」上一状态里本集已经完成的条目，必须逐字引用上一状态的原文（不要改写）。
- 「完成」的判定标准：本集是否得出了【能回答该目标自身问题的确定结论】。
  - 正例：目标「解读便签纸内容」，本集得出「便签指向 B7-L3-S5 货位」→ 已能回答"便签写了什么"，算完成。
  - 反例：目标「追查幕后主使」，本集只得到「周启明重启了服务器」这条线索，仍未回答"幕后主使是谁"→ 不算完成，不要输出。
  - 关键：只要「这个目标要查的事」有了确定答案就算完成，不需要等到整个故事真相大白。不要把"主线还没结局"当作"所有子目标都未完成"的理由。
  - 重要反例：「人物决定/计划去做某事」不等于目标完成。本集出现「决定去查监控」「决定去找某人」「商量好下一步怎么做」，是「建立行动计划」，应记入 new_obligations 的 PLAN，不是「目标已得出确定结论」。只有「查出了什么结果」「找到了谁」「确认了什么事实」才算 resolved。严禁把 PLAN 的内容同时写进 resolved_goals。
- resolved_goals：本集已得出确定答案的目标。resolved_unknown：本集已揭晓答案的未知。retracted_facts：本集被推翻的旧事实。
- resolved_unknown 的「揭晓」标准比 goal 更严格：未知项通常是「是/否、谁、为什么」这类疑问，只有当本集给出了【对该疑问本身的直接明确答案】才算揭晓。例如「周启明是否参与调包」→ 必须出现「周启明确实（未）参与调包」这种确定结论；若本集只是获得相关线索（如「周启明还活着」「持有钥匙」），但并未直接回答「是否参与」这个疑问，则不算揭晓，不要输出。
- 拿不准「本集是否已经回答了该目标的问题」时，宁可保守不清除；但已经查出明确结论的小目标，不要因为故事还没结局就留着。没有失效条目时用空数组。
- new_obligations 只记录本集剧本实际新建立的六类叙事债务，每集最多 2 条，宁缺毋滥。所有条目都输出 deadline 字段：仅 DEADLINE 填剧本中的明确期限原文，其余类型填 null：
  - FORESHADOW（伏笔）：剧本埋下、但本集未解释其意义、留待后续揭示的细节——重复出现的物件、反常动作或台词、人物刻意回避或遮住的信息、被特写强调却未交代来源的物品。它不要求是明确疑问。例：反复出现的旧钥匙、物品上的不明血迹、人物刻意遮住的照片、一句意味深长却未展开的台词。
  - MYSTERY（谜题）：剧本明确提出、有待回答的「谁/为什么/是什么」式疑问。例：谁执行了修改指令、幽灵卡从哪来。
  - PROMISE（承诺）：人物对他人或自己作出的明确保证、誓言、约定（常见「会/不会/一定/答应/保证/发誓」等措辞），未来需用行动兑现或食言承担后果。例：我保证三天内找到他、我答应不会再隐瞒、我会把真相告诉父亲。
  - PLAN（计划）：人物已经形成且决定执行的多步行动方案，未来需要看到执行、变更、失败或放弃。例：先转移羊群再封闭旧牧道、按分工排练并在周末合成。临时待办或单步动作不算 PLAN。
  - DEADLINE（期限）：剧情明确建立了一个不可无限拖延的时间窗口，并存在到期后果。例：春播前必须完成迁徙、名单截止前必须决定补位人选。deadline 填剧本中的原始期限表述；若没有明确期限则不要标 DEADLINE。
  - RELATIONSHIP_DEBT（关系债）：人物间因伤害、亏欠、救助、隐瞒或交换而形成、未来必须通过道歉、补偿、解释、信任回应或关系决断来结清的债。普通情绪波动、争吵或关系变化不算。
  - 关键区分：伏笔是"待解释的细节"，谜题是"待回答的问题"，承诺是"人物的保证"，计划是"已决定执行的方案"，期限是"有到期后果的时间窗口"，关系债是"未来必须回应的人际亏欠"。同一事件只选最主要的一类，不要重复登记。不得把 episode goal、hard anchors、作者要求或普通待办改写成义务。
- resolved_obligations：逐条对照「当前开放叙事义务」中的每一条，只有【本集真的发生了该义务 required_payoff 要求的那个具体结果】才兑现，并引用其 description 原文（可轻微改写但语义必须一致）：
  - FORESHADOW：细节的叙事意义被直接揭示；MYSTERY：问题得到明确答案；PROMISE：承诺行动已完成，或人物明确撤回并承担后果；PLAN：计划完成、明确失败、被替代或正式放弃；DEADLINE：【期限已到达且到期后果已落地】才能兑现，期限未到一律不兑现；RELATIONSHIP_DEBT：道歉、补偿、解释、信任回应或关系决断已经发生。
  - 铁律：剧情重大进展、关系转折、人物和解、临近到期，都不等于义务兑现。逐条问自己「这条义务要的那个具体结果，本集是否真的发生了」。只提供线索、提到义务、部分推进、重复计划，都不算兑现。
  - 每集最多兑现 2 条，只兑现最确凿的 1-2 条；拿不准的一律不兑现。
- 状态记忆只能记录故事世界内已经发生、确认或尚待执行的内容，不得记录作者要求、hard anchors、实验目的或写作策略。
- confirmed_facts 只能记录肯定式世界事实，不得写入“本集必须、结尾必须、不能确认、不得出现、需要保留解释”等规范性文本。
- current_goals 只能记录人物在故事世界内尚待执行的具体任务，不得写入“保持证据边界、维持悬念、避免锁定嫌疑、保留分支”等创作目标。
- relationship_state 只记录信任、冲突、依赖、承诺和关系边界的实际变化，不记录临时调查分工、谁查哪条线、谁负责哪份资料或场景调度。
- 临时任务安排如确需继承，应写入 current_goals；不得把 episode plan 的原句改写后写回状态。
- 九个状态字段必须全部出现；没有新增时使用空数组，绝不能把 state_delta 输出为数组、字符串或 null。
- character_state 和 relationship_state 每个发生变化的主体最多写 2 项；其他每个字段最多写 4 项；每项不超过 35 个中文字符。
- possible conflicts 不在本 schema 中，不要额外输出；secondary_move_ids 最多 2 项；move_evidence 不超过 60 字。
- card_feedback 只评估「本集受控试用卡」：每张试用卡恰好一项且 memory_id 必须原样复制。adopted=true 表示本集实质采用了该卡的核心意图——应对了其适用情境并实现了其预期状态变化（expected_state_change），evidence 引用剧本中体现该采用/变化的片段；不要求逐字执行卡中的具体步骤，但仅主题相似、Move 相同或结果碰巧相似不算采用。harm_flags 只能取 STATE_CONFLICT/ANCHOR_VIOLATION/CAUSAL_BREAK/MECHANISM_REPETITION/CARD_ECHO/UNSUPPORTED_INVENTION；没有试用卡时输出空数组。
- 仅当「本集经验形成上下文」的 candidate_generation_allowed=true 时才允许生成 writeback_candidate；否则必须输出 null。
- writeback_candidate 只总结本集已经成功实现、现有经验卡未直接覆盖且可跨故事复用的结构操作；普通推进、仅换人物地点道具、无法写出两步 operator、含人物名/地点/专有道具时输出 null。quality_score 与 confidence 必须是 0.0 到 1.0 之间的小数（1 表示最高，不是 10 分制也不是 100 分制）。applicable_state_features 必须从「本集经验形成上下文」的 state_tags 逐字选择 2-3 项，不得自造标签；每个文本字段不超过 45 字，operator_steps 2-3 项，risks/abort_conditions 最多 2 项。
- 未确认推测不进入 confirmed_facts。最外层直接输出上述对象，不要包在 result、output 或 data 字段中。
- 整个响应控制在 2600 个中文字符以内，优先保证 JSON 完整闭合和必需字段齐全。"""
    if exclude_relationship_debt:
        prompt = (
            prompt
            .replace("FORESHADOW|MYSTERY|PROMISE|PLAN|DEADLINE|RELATIONSHIP_DEBT", "FORESHADOW|MYSTERY|PROMISE|PLAN|DEADLINE")
            .replace("六类叙事债务", "五类叙事债务")
            .replace(
                "  - RELATIONSHIP_DEBT（关系债）：人物间因伤害、亏欠、救助、隐瞒或交换而形成、未来必须通过道歉、补偿、解释、信任回应或关系决断来结清的债。普通情绪波动、争吵或关系变化不算。",
                "  - RELATIONSHIP_DEBT 由独立人物关系记忆处理；本输出禁止标注该类型，也不得用 PLAN 代替关系债。",
            )
            .replace(
                "期限是\"有到期后果的时间窗口\"，关系债是\"未来必须回应的人际亏欠\"。同一事件只选最主要的一类",
                "期限是\"有到期后果的时间窗口\"。关系债不进入本义务池，也不得改写为 PLAN。同一事件只选最主要的一类",
            )
            .replace("；RELATIONSHIP_DEBT：道歉、补偿、解释、信任回应或关系决断已经发生。", "")
            .replace(
                "- current_goals 只能记录人物在故事世界内尚待执行的具体任务，不得写入“保持证据边界、维持悬念、避免锁定嫌疑、保留分支”等创作目标。",
                "- current_goals 只能记录人物在故事世界内尚待执行的非关系债任务；待道歉、解释、补偿、获得原谅或信任回应不得写入 current_goals，由独立人物关系记忆处理。不得写入“保持证据边界、维持悬念、避免锁定嫌疑、保留分支”等创作目标。",
            )
            .replace(
                "- relationship_state 只记录信任、冲突、依赖、承诺和关系边界的实际变化，不记录临时调查分工、谁查哪条线、谁负责哪份资料或场景调度。",
                "- relationship_state 只记录本集已经发生的信任、冲突、依赖、承诺和关系边界变化；尚待解释、道歉、补偿、回应或结清的关系债不得写入普通状态，由独立人物关系记忆处理。不记录临时调查分工、谁查哪条线、谁负责哪份资料或场景调度。",
            )
        )
    return prompt


def build_compact_extraction_prompt(*, script: str, move_ids: list[str], validation_error: str, trial_cards: list[dict[str, Any]] | None = None, experience_context: dict[str, Any] | None = None) -> str:
    return f"""你是 JSON 信息抽取器。上一响应因“{validation_error}”无效。只读取下面的当前集剧本，不参考或复述此前剧情。只输出一个完整、合法、直接闭合的 JSON 对象，禁止 Markdown 和解释。

当前集剧本：
{script}

本集经验形成上下文：{block(experience_context or {})}
本集受控试用卡：{block(trial_cards or [])}
允许的 move_id：{block(move_ids)}

严格输出：
{{"state_delta":{{"character_state":[],"relationship_state":[],"known_information":[],"unknown_information":[],"confirmed_facts":[],"unresolved_threads":[],"resources_and_evidence":[],"current_goals":[],"timeline":[]}},"resolved_goals":[],"resolved_unknown":[],"retracted_facts":[],"new_obligations":[],"resolved_obligations":[],"primary_move_id":"从允许列表选一个","secondary_move_ids":[],"move_evidence":"不超过40字","state_constraint_count":0,"state_consistency_issues":[],"card_feedback":[],"writeback_candidate":null}}

只提取本集明确新增或改变的故事事实。不得记录作者要求、连续性边界、实验目的或写作策略；confirmed_facts 不写“必须、不能确认、不得出现”等规则；current_goals 只写人物尚待执行的具体任务；relationship_state 只写实际关系变化，不写临时调查分工。若存在受控试用卡，card_feedback 必须逐卡输出并原样复制 memory_id；adopted 表示本集实质实现了该卡的预期状态变化（效果归因，不要求逐字执行步骤），evidence 引用剧本逐字片段。修复重试时允许 writeback_candidate=null，优先保证状态与 card_feedback 完整。每个状态字段最多 3 项，每项不超过 30 字；整个响应不超过 1700 个中文字符。"""


def build_evaluation_prompt(*, story: dict[str, Any], plan: dict[str, Any], previous_state: dict[str, Any], script: str) -> str:
    return f"""你是以“毒舌、零容忍”著称的资深剧本评审，进行盲评：不得猜测实验条件。评估范围限定在本集质量与给定上下文的衔接；跨集前史是否自洽不在职责内。任何评分必须有具体场次/台词/描述支撑，禁止“感觉分”，禁止保守居中。

故事：{block({key: story[key] for key in ('title', 'genre', 'world_setting', 'series_goal', 'characters')})}\n上一状态：{block(previous_state)}\n当前集要求：{block(plan)}\n剧本：\n{script}

对以下 8 个维度按统一扣分制打分：每维满分 100，从 100 起评；逐项发现问题按量化参照扣分，亮点可加分；每维得分 = max(0, min(100, 100 + 各条 points 之和))，输出整数。

【评分校准纪律】
- 85 分以上必须在该维 details 中列出至少一条具体亮点及文本证据，否则不得高于 84。
- 无问题也无亮点的平庸稿应在 70-80 区间；AI 生成单集常见区间为 55-80；90 分以上应极少出现。
- 命中量化阈值时按区间给扣分，不得因“整体还行”而模糊降档或机械居中。

【维度与扣分参照】（points 扣分为负、亮点为正，可按严重程度在区间内连续取值）
1. dialogue_quality（台词质量）：AI味/模型化表达（总结腔、说理替代回应、句式机械同质、“不是…而是…”升华句式）占比>50% 扣30-40，30%-50% 扣20-30，10%-30% 扣8-20；人物语言辨识度不足：主要角色完全混同扣20-25，关键角色风格相近扣10-18；信息量失衡（注水/设定硬塞/前史硬塞）每条扣5-10；重场戏情绪曲线平扣10-20；亮点：语言风格化设计+3-8，可传播金句+2-5。
2. scene_structure（场次结构）：低效/无效场次（无新增信息、无人物变化、无情绪推进）每场扣8-15；节奏拖沓、高潮缺失或位置失当扣10-20；场间衔接生硬每处扣3-8；集末钩子缺失或过弱扣8-12；亮点：场次强逻辑咬合、情绪曲线清晰+3-8。
3. visual_adaptability（视觉转化）：动作/场景描写占比<20% 扣20-30，20%-35% 扣8-15；非视觉化描述（心理活动、上帝视角议论、文学化比喻、总结性陈述）每处扣3-8且本项累计不超过25；关键情节动作指示模糊或缺失扣8-15；亮点：画面感强、有完整可视名场面+3-8。
4. state_consistency（状态一致性）：与上一状态直接矛盾每处扣15-25；人物状态无依据漂移（性格/关系/立场突变）每处扣8-15；擅自确认上一状态标记为未知的信息每处扣10-20。
5. anchor_satisfaction（戏剧任务达成）：本集核心目标未达成扣25-40，部分达成扣8-15；越过集末边界（提前揭示禁止确认的信息、越权行动）每条扣15-30。
6. causal_continuity（因果连续性）：场景间无因果连接、事件并置拼贴扣15-25；关键行动无后果或动机断裂每处扣8-15。
7. mechanism_novelty（机制新颖度）：核心推进机制机械复述常见套路或与本集情境贴合度低扣10-20；机制与前序集高度雷同、无新的信息结构扣8-15；亮点：机制组合新颖且自洽+3-10。
8. rule_echo_avoidance（规则复述规避）：角色台词复述幕后边界（“不能确认”“证据不足不能下结论”“不能越权”等结论性规则口号）每处扣8-15。

【输出格式】只输出合法 JSON 对象，不要 Markdown：
{{"dialogue_quality":0,"scene_structure":0,"visual_adaptability":0,"state_consistency":0,"anchor_satisfaction":0,"causal_continuity":0,"mechanism_novelty":0,"rule_echo_avoidance":0,"details":{{"dialogue_quality":[{{"type":"问题","item":"问题名","evidence":"具体场次/台词/描述引用","points":-12}}],"scene_structure":[],"visual_adaptability":[],"state_consistency":[],"anchor_satisfaction":[],"causal_continuity":[],"mechanism_novelty":[],"rule_echo_avoidance":[]}},"justifications":{{"dialogue_quality":"一句结论","scene_structure":"","visual_adaptability":"","state_consistency":"","anchor_satisfaction":"","causal_continuity":"","mechanism_novelty":"","rule_echo_avoidance":""}},"rule_echo_spans":[],"violations":[]}}

- details 必须列出全部 8 个维度；每条含 type（问题/亮点）、item、evidence、points；无问题无亮点给空数组。
- 各维得分必须等于 100 加该维 details 全部 points 之和并夹在 0-100。
- rule_echo_spans 逐字摘录疑似规则复述台词，没有则为空数组；violations 列其他硬违规，没有则为空数组。"""


def build_branch_prompt(*, story: dict[str, Any], plan: dict[str, Any], state: dict[str, Any], count: int, move_ids: list[str]) -> str:
    return f"""基于故事状态生成 {count} 个满足集末边界、因果可行且机制异质的后续分支。边界只用于内部检查，不要写进人物对白。只输出 JSON 数组，不写剧本。\n故事：{block(story)}\n状态：{block(state)}\n集要求：{block(plan)}\nMove：{block(move_ids)}\n每项格式：{{"branch_id":"B1","move_id":"...","outline":"...","constraint_check":[]}}。"""
