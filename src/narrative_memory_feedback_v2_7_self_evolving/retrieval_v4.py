"""Pure V4 retrieval: episode function × state fit × invariant safety × repair gates."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

POOL_PATH = Path(__file__).resolve().parent / "data" / "memory_pools_v4.json"
ALL_FUNCTIONS = ("INITIATE", "PROGRESS", "TEST", "CONSEQUENCE", "PRESERVE", "CONSOLIDATE", "RECOVER", "CLOSE")
MIN_STATE_SCORE = 2

# 义务类型 → 机制族 token 亲和映射（token 按下划线分词，避免 SOURCE/RESOURCE 这类子串误匹配）。
# favor：这类义务兑现/推进需要的结构操作；avoid：容易破坏/改写这类义务的结构操作。
OBLIGATION_FAVORED_TOKENS: dict[str, set[str]] = {
    "FORESHADOW": {"DEFER", "PRESERVE", "UNEARNED", "REVEAL"},
    "MYSTERY": {"SOURCE", "EXPLANATION", "EXPLANATIONS", "VERIFICATION", "UNEARNED"},
    "PROMISE": {"COMMITMENT", "HANDOFF"},
    "PLAN": {"SEQUENCE", "METHOD", "MILESTONE", "REDESIGN"},
    "DEADLINE": {"DEADLINE", "CLOSURE", "CLOSED", "WINDOW"},
    "RELATIONSHIP_DEBT": {"RELATIONSHIP", "TRUST", "RECONCILIATION", "RELIABILITY"},
}
OBLIGATION_AVOIDED_TOKENS: dict[str, set[str]] = {
    # S3 义务记忆的关键副作用：关系债被改写成计划/承诺/交接，损害人际张力。
    "RELATIONSHIP_DEBT": {"COMMITMENT", "SEQUENCE", "HANDOFF", "MILESTONE", "PROCEDURE", "PLAN"},
}


def obligation_affinity(mechanism_family: str, open_obligations: Iterable[dict[str, Any]] | None) -> int:
    """计算一张卡相对当前开放义务的亲和度（favor 加分、avoid 减分）。"""
    if not open_obligations:
        return 0
    tokens = set(mechanism_family.split("_"))
    affinity = 0
    for obligation in open_obligations:
        otype = str(obligation.get("type", ""))
        if otype in OBLIGATION_FAVORED_TOKENS and tokens & OBLIGATION_FAVORED_TOKENS[otype]:
            affinity += 1
        if otype in OBLIGATION_AVOIDED_TOKENS and tokens & OBLIGATION_AVOIDED_TOKENS[otype]:
            affinity -= 2
    return affinity

FUNCTION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CLOSE", ("最终", "结局", "终局", "最后一次", "告别", "收束全", "关闭全部")),
    ("CONSOLIDATE", ("定型", "固定", "写成", "正式", "交接", "落实", "形成责任", "确定上限", "谈定", "重排")),
    ("CONSEQUENCE", ("损失", "代价", "失去", "后果", "受损", "不可挽回", "永久不可", "关闭路径")),
    ("RECOVER", ("修复", "重建", "改法", "替代方案", "重新安排", "恢复行动", "纠正", "返工")),
    ("TEST", ("检验", "核验", "验证", "测试", "评估", "复盘", "清点", "盘点", "核算", "比对")),
    ("PRESERVE", ("保持悬置", "保留未知", "不揭示", "暂不决定", "维持边界", "不得解决")),
    ("INITIATE", ("建立", "发起", "接下", "首次", "第一次", "形成有限合作", "打开入口")),
)

INVARIANT_PATTERNS: dict[str, tuple[str, ...]] = {
    "ESTABLISHED_LOSS": (
        "永久损失", "不得恢复", "不可逆", "不得逆转", "已失去", "已经失去", "无法保存",
        "永久痕迹", "不得追回", "不得复原", "不得康复", "不得自动恢复", "不能恢复",
    ),
    "INFORMATION_OPEN": (
        "不得揭示", "不揭示", "不得确认", "不能确认", "不足以确认", "不被确认为", "不得被确认",
        "仍然未知", "仍不得被确认", "不可知", "留作未解", "保持悬置", "不得解释", "不得追回",
        "不能回答", "不能指认", "不得指认", "不得查明", "仍未查清", "关系未解",
        "不得锁定任何人为", "不能确认最终",
    ),
    "OPTION_OPEN": (
        "不得决定", "不得直接决定", "不得当场接受", "不得完成最终方案", "本集不得完成",
        "本集不做", "暂不决定", "不能同时完成", "不得锁定任何人为", "不做来年最终方案",
    ),
    "CAPABILITY_CEILING": (
        "能力不得恢复", "恢复独立生活能力", "能力范围", "技能范围", "职责范围", "上限", "只能承担",
        "只能提供", "不能承担", "不得承担", "不得独立", "不得超出", "不能超出", "有限参与",
        "只能确定", "只能证明", "只能确认", "只能使用", "不得全面", "不可独立",
    ),
    "RELATIONSHIP_NONRESOLUTION": (
        "不得彻底和解", "不得达成三方彻底和解", "不得通过一次谈话彻底修复关系", "不得被视作同意",
        "不得把沉默视作同意", "不得被确认为同意或反对", "不得被解释为明确同意", "边界继续有效",
        "处置权继续有效", "保留分歧", "未解决的分歧", "不得彻底推翻协作机制",
    ),
    "RESOURCE_CEILING": (
        "不得新增", "不得出现新增", "不得提供额外", "不得外部救援", "外部救助", "外部救援",
        "只使用", "必须只使用", "现金缺口", "资源不足", "不得自动恢复", "资源自动恢复",
        "不得出现意外", "不得引入新", "不得引入此前未出现", "不能重新获得", "不得无偿",
    ),
    "AGENCY_OWNERSHIP": (
        "不得由", "不能由", "不能替", "不得替", "自行承担", "不能代做", "不得暗中替", "不能以辞",
        "不能单独裁定", "不得替他补完", "不得替家庭决定", "不得替班社决定", "不得替其他人决定",
    ),
    "CAUSAL_CONTINUITY": (
        "承接", "必须来自", "建立在", "源于", "沿用", "继续有效", "必须继承", "必须使用",
        "必须参考", "必须由", "必须符合", "必须进入后续", "必须计入", "累积而来",
    ),
    "OPEN_OBLIGATION": (
        "必须保留", "留作未解", "进入后续", "下一集", "后续使用", "后续核查", "后续状态",
        "后续任务", "长期代价", "长期缺口", "未解问题", "留下尚未解决", "仍需承担",
    ),
}

BLOCKER_MECHANISMS_BY_INVARIANT: dict[str, set[str]] = {
    "ESTABLISHED_LOSS": {"BLOCK_CONSEQUENCE_RESET"},
    "INFORMATION_OPEN": {"BLOCK_UNEARNED_REVEAL"},
    "OPTION_OPEN": {"BLOCK_FAKE_CHOICE"},
    "CAPABILITY_CEILING": {"BLOCK_CAPABILITY_JUMP"},
    "RELATIONSHIP_NONRESOLUTION": {"BLOCK_INSTANT_RECONCILIATION"},
    "RESOURCE_CEILING": {"BLOCK_EXTERNAL_RESCUE"},
    "AGENCY_OWNERSHIP": {"BLOCK_EXTERNAL_RESCUE"},
    "CAUSAL_CONTINUITY": set(),
    "OPEN_OBLIGATION": {"BLOCK_UNEARNED_REVEAL", "BLOCK_OBLIGATION_DROP"},
}


@dataclass(frozen=True)
class RetrievalDecision:
    selected_card: dict[str, Any] | None
    active_blockers: list[dict[str, Any]]
    episode_function: str
    function_evidence: str
    protected_invariants: list[str]
    reason: str
    top_score: int | None
    eligible_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_pool(path: Path = POOL_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def phase_for_episode(index: int, total_episodes: int) -> str:
    ratio = index / max(total_episodes, 1)
    if ratio < 0.15:
        return "setup"
    if ratio < 0.35:
        return "expansion"
    if ratio < 0.65:
        return "escalation"
    return "closure"


def infer_episode_function(job: dict[str, Any], total_episodes: int = 40) -> tuple[str, str]:
    explicit = str(job.get("episode_function", "")).upper()
    if explicit in ALL_FUNCTIONS:
        return explicit, "explicit_episode_function"

    index = int(job.get("episode_index", 0))
    goal = str(job.get("episode_goal", ""))
    if index >= max(total_episodes - 2, 0):
        return "CLOSE", "final_episode_window"
    for function_id, keywords in FUNCTION_KEYWORDS:
        match = next((keyword for keyword in keywords if keyword in goal), None)
        if match:
            return function_id, f"goal_keyword:{match}"
    if index == 0:
        return "INITIATE", "first_episode_fallback"
    return "PROGRESS", "default_progress"


def infer_protected_invariants(hard_anchors: Iterable[str]) -> tuple[set[str], dict[str, list[str]]]:
    anchors = [str(anchor) for anchor in hard_anchors]
    evidence: dict[str, list[str]] = {}
    for invariant_id, patterns in INVARIANT_PATTERNS.items():
        hits = [anchor for anchor in anchors if any(pattern in anchor for pattern in patterns)]
        if hits:
            evidence[invariant_id] = hits
    return set(evidence), evidence


def active_blockers(
    cards: list[dict[str, Any]],
    invariants: set[str],
    episode_function: str,
    diagnostic_tags: set[str],
) -> list[dict[str, Any]]:
    mechanisms: set[str] = set()
    for invariant_id in invariants:
        mechanisms.update(BLOCKER_MECHANISMS_BY_INVARIANT.get(invariant_id, set()))
    if episode_function == "PRESERVE":
        mechanisms.add("BLOCK_FORCED_ESCALATION")
    if "exposition_heavy" in diagnostic_tags:
        mechanisms.add("BLOCK_EXPOSITION_AS_ACTION")
    return [
        card for card in cards
        if card["card_type"] == "ANTI_PATTERN"
        and card.get("status") != "retired"
        and episode_function in card.get("episode_functions", [])
        and card.get("mechanism_family") in mechanisms
    ]


def _cooldown_ok(card: dict[str, Any], recent_card_ids: list[str], episode_index: int, cooldown_state: dict[str, int]) -> bool:
    memory_id = card["memory_id"]
    last_used = cooldown_state.get(memory_id, -999)
    cooldown = int(card.get("cooldown_episodes", 0))
    return memory_id not in recent_card_ids[-1:] and episode_index - last_used >= cooldown


def _strategy_candidates(
    cards: list[dict[str, Any]],
    episode_function: str,
    phase: str,
    state_tags: set[str],
    invariants: set[str],
    trajectory_declining: bool,
    recent_mechanisms: set[str],
    recent_card_ids: list[str],
    episode_index: int,
    cooldown_state: dict[str, int],
    open_obligations: Iterable[dict[str, Any]] | None = None,
) -> list[tuple[dict[str, Any], int, int]]:
    candidates: list[tuple[dict[str, Any], int, int]] = []
    for card in cards:
        if card["card_type"] != "STRATEGY" or card.get("status") == "retired":
            continue
        functions = card.get("episode_functions", [])
        if episode_function not in functions or phase not in card.get("applicable_phases", []):
            continue
        requires_all = set(card.get("requires_all", []))
        requires_any = set(card.get("requires_any", []))
        if requires_all - state_tags or set(card.get("forbids_any", [])) & state_tags:
            continue
        if set(card.get("conflicts_with_invariants", [])) & invariants:
            continue
        if not _cooldown_ok(card, recent_card_ids, episode_index, cooldown_state):
            continue
        state_score = len(requires_all) + len(requires_any & state_tags)
        if state_score < MIN_STATE_SCORE:
            continue
        function_score = 2 if functions[0] == episode_function else 1
        diversity_score = 1 if card.get("mechanism_family") not in recent_mechanisms else 0
        obligation_score = obligation_affinity(card.get("mechanism_family", ""), open_obligations)
        score = state_score * 2 + function_score + diversity_score + obligation_score
        move = card.get("primary_move_id") or ""
        if trajectory_declining:
            if move.endswith("_LOSS") or move.endswith("_REFRAME"):
                score += 1
            elif move.endswith("_GAIN"):
                score -= 1
        candidates.append((card, score, state_score))
    return sorted(candidates, key=lambda item: (-item[1], item[0]["memory_id"]))


def _repair_candidates(
    cards: list[dict[str, Any]],
    episode_function: str,
    phase: str,
    state_tags: set[str],
    diagnostic_tags: set[str],
    invariants: set[str],
    recent_card_ids: list[str],
    episode_index: int,
    cooldown_state: dict[str, int],
) -> list[tuple[dict[str, Any], int]]:
    candidates: list[tuple[dict[str, Any], int]] = []
    available_tags = state_tags | diagnostic_tags
    for card in cards:
        if card["card_type"] != "REPAIR" or card.get("status") == "retired":
            continue
        if episode_function not in card.get("episode_functions", []) or phase not in card.get("applicable_phases", []):
            continue
        requires_all = set(card.get("requires_all", []))
        requires_any = set(card.get("requires_any", []))
        if requires_all - available_tags:
            continue
        if requires_any and not (requires_any & available_tags):
            continue
        if set(card.get("forbids_any", [])) & available_tags:
            continue
        if set(card.get("conflicts_with_invariants", [])) & invariants:
            continue
        matched = set(card.get("repair_triggers", [])) & diagnostic_tags
        if not matched or not _cooldown_ok(card, recent_card_ids, episode_index, cooldown_state):
            continue
        score = len(matched) * 2 + len(requires_all) + len(requires_any & available_tags)
        candidates.append((card, score))
    return sorted(candidates, key=lambda item: (-item[1], item[0]["memory_id"]))


def select_v4_card(
    cards: list[dict[str, Any]],
    job: dict[str, Any],
    state_tags: set[str],
    diagnostic_tags: set[str] | None = None,
    trajectory_declining: bool = False,
    recent_card_ids: list[str] | None = None,
    cooldown_state: dict[str, int] | None = None,
    total_episodes: int = 40,
    open_obligations: Iterable[dict[str, Any]] | None = None,
) -> RetrievalDecision:
    diagnostics = set(diagnostic_tags or set())
    recent_ids = list(recent_card_ids or [])
    cooldowns = dict(cooldown_state or {})
    episode_index = int(job.get("episode_index", 0))
    episode_function, function_evidence = infer_episode_function(job, total_episodes)
    invariants, _ = infer_protected_invariants(job.get("hard_anchors", []))
    blockers = active_blockers(cards, invariants, episode_function, diagnostics)
    phase = phase_for_episode(episode_index, total_episodes)

    repairs = _repair_candidates(
        cards, episode_function, phase, state_tags, diagnostics, invariants,
        recent_ids, episode_index, cooldowns,
    )
    if repairs:
        selected, score = repairs[0]
        return RetrievalDecision(selected, blockers, episode_function, function_evidence, sorted(invariants), "gated_repair", score, len(repairs))

    card_by_id = {card["memory_id"]: card for card in cards}
    recent_mechanisms = {
        card_by_id[memory_id].get("mechanism_family", "")
        for memory_id in recent_ids[-3:]
        if memory_id in card_by_id
    }
    candidates = _strategy_candidates(
        cards, episode_function, phase, state_tags, invariants, trajectory_declining,
        recent_mechanisms, recent_ids, episode_index, cooldowns,
        open_obligations=open_obligations,
    )
    if not candidates:
        return RetrievalDecision(None, blockers, episode_function, function_evidence, sorted(invariants), "safe_abstain_no_eligible_strategy", None, 0)
    selected, score, _ = candidates[0]
    return RetrievalDecision(selected, blockers, episode_function, function_evidence, sorted(invariants), "function_state_invariant_top1", score, len(candidates))


def select_v4_trial_card(
    cards: list[dict[str, Any]],
    job: dict[str, Any],
    state_tags: set[str],
    recent_card_ids: list[str] | None = None,
    cooldown_state: dict[str, int] | None = None,
    total_episodes: int = 40,
) -> RetrievalDecision:
    """为隔离候选池选择一张试用卡。

    试用仍严格满足状态前提、禁用条件和 invariant；功能/阶段匹配仅作为排序加分，
    不作为硬门槛——否则“INITIATE/setup 形成、但状态前提直到 expansion 才重现”的
    候选会永远没有复验机会（受控试用的意义正是跨上下文验证其泛化与副作用）。
    """
    recent_ids = list(recent_card_ids or [])
    cooldowns = dict(cooldown_state or {})
    episode_index = int(job.get("episode_index", 0))
    episode_function, function_evidence = infer_episode_function(job, total_episodes)
    invariants, _ = infer_protected_invariants(job.get("hard_anchors", []))
    phase = phase_for_episode(episode_index, total_episodes)
    candidates: list[tuple[dict[str, Any], int]] = []
    for card in cards:
        if card.get("card_type") != "STRATEGY" or card.get("lifecycle_status") not in {"CANDIDATE", "TRIAL"}:
            continue
        function_match = episode_function in card.get("episode_functions", [])
        phase_match = phase in card.get("applicable_phases", [])
        requires_all = set(card.get("requires_all", []))
        requires_any = set(card.get("requires_any", []))
        if requires_all - state_tags or (requires_any and not requires_any & state_tags):
            continue
        if set(card.get("forbids_any", [])) & state_tags:
            continue
        if set(card.get("conflicts_with_invariants", [])) & invariants:
            continue
        if not _cooldown_ok(card, recent_ids, episode_index, cooldowns):
            continue
        state_score = len(requires_all) + len(requires_any & state_tags)
        if state_score < MIN_STATE_SCORE:
            continue
        compatibility_score = (2 if function_match else 0) + (1 if phase_match else 0)
        candidates.append((card, state_score * 2 + compatibility_score))
    candidates.sort(key=lambda item: (-item[1], item[0]["memory_id"]))
    if not candidates:
        return RetrievalDecision(
            None, [], episode_function, function_evidence, sorted(invariants),
            "candidate_trial_abstain", None, 0,
        )
    selected, score = candidates[0]
    return RetrievalDecision(
        selected, [], episode_function, function_evidence, sorted(invariants),
        "candidate_trial_state_compatible", score, len(candidates),
    )


def select_v4_gate_only(cards: list[dict[str, Any]], job: dict[str, Any], total_episodes: int = 40) -> RetrievalDecision:
    """功能标注 + 门控（无 strategy 卡）：用于在干净生命周期上分解门控的独立贡献。"""
    episode_function, function_evidence = infer_episode_function(job, total_episodes)
    invariants, _ = infer_protected_invariants(job.get("hard_anchors", []))
    blockers = active_blockers(cards, invariants, episode_function, set())
    return RetrievalDecision(
        None, blockers, episode_function, function_evidence,
        sorted(invariants), "gate_only_no_strategy", None, 0,
    )


def select_v4_gate_repair(
    cards: list[dict[str, Any]],
    job: dict[str, Any],
    state_tags: set[str],
    diagnostic_tags: set[str] | None = None,
    *,
    recent_card_ids: list[str] | None = None,
    cooldown_state: dict[str, int] | None = None,
    total_episodes: int = 40,
) -> RetrievalDecision:
    """功能标注 + 门控 + repair（无 strategy 卡）：验证 repair 卡的贡献。

    repair 卡通过 diagnostic_tags 里的触发信号激活（优先于 strategy），
    无 repair 触发时不塞 strategy 卡（保持门控 only 的克制）。
    """
    diagnostics = set(diagnostic_tags or set())
    recent_ids = list(recent_card_ids or [])
    cooldowns = dict(cooldown_state or {})
    episode_index = int(job.get("episode_index", 0))
    episode_function, function_evidence = infer_episode_function(job, total_episodes)
    invariants, _ = infer_protected_invariants(job.get("hard_anchors", []))
    blockers = active_blockers(cards, invariants, episode_function, diagnostics)
    phase = phase_for_episode(episode_index, total_episodes)
    repairs = _repair_candidates(
        cards, episode_function, phase, state_tags, diagnostics, invariants,
        recent_ids, episode_index, cooldowns,
    )
    if repairs:
        selected, score = repairs[0]
        return RetrievalDecision(selected, blockers, episode_function, function_evidence, sorted(invariants), "gated_repair", score, len(repairs))
    return RetrievalDecision(None, blockers, episode_function, function_evidence, sorted(invariants), "gate_only_no_repair_no_strategy", None, 0)


def select_v4_strategy_only(
    cards: list[dict[str, Any]],
    job: dict[str, Any],
    state_tags: set[str],
    *,
    trajectory_declining: bool = False,
    recent_card_ids: list[str] | None = None,
    cooldown_state: dict[str, int] | None = None,
    total_episodes: int = 40,
) -> RetrievalDecision:
    """只选 strategy 卡（无门控、无功能标注）：用于分解 strategy 卡的独立贡献。

    episode_function / invariants 仍在内部用于卡的适用性过滤，但不注入 prompt。
    """
    recent_ids = list(recent_card_ids or [])
    cooldowns = dict(cooldown_state or {})
    episode_index = int(job.get("episode_index", 0))
    episode_function, _ = infer_episode_function(job, total_episodes)
    invariants, _ = infer_protected_invariants(job.get("hard_anchors", []))
    phase = phase_for_episode(episode_index, total_episodes)
    card_by_id = {card["memory_id"]: card for card in cards}
    recent_mechanisms = {
        card_by_id[memory_id].get("mechanism_family", "")
        for memory_id in recent_ids[-3:]
        if memory_id in card_by_id
    }
    candidates = _strategy_candidates(
        cards, episode_function, phase, state_tags, invariants,
        trajectory_declining, recent_mechanisms, recent_ids,
        episode_index, cooldowns,
    )
    if not candidates:
        return RetrievalDecision(None, [], "", "", [], "safe_abstain_no_eligible_strategy", None, 0)
    selected, score, _ = candidates[0]
    return RetrievalDecision(selected, [], "", "", [], "strategy_only_top1", score, len(candidates))


def select_v4_function_only(job: dict[str, Any], total_episodes: int = 40) -> RetrievalDecision:
    """功能标注（无门控、无卡）：用于分解功能标注的独立贡献。"""
    episode_function, function_evidence = infer_episode_function(job, total_episodes)
    return RetrievalDecision(
        None, [], episode_function, function_evidence,
        [], "function_only_no_gate", None, 0,
    )


def v4_experience_block(decision: RetrievalDecision, *, lenient: bool = False, max_blockers: int | None = None, include_steps: bool = True) -> str:
    lines = [f"本集功能：{decision.episode_function}"] if decision.episode_function else []
    blockers = list(decision.active_blockers)
    if max_blockers is not None:
        blockers = blockers[:max_blockers]
    if blockers:
        if lenient:
            # 宽松门控：只留「禁止结果」硬约束，去掉指令化的检查动作/中止条件，减少对创作行动的束缚。
            lines.append("必须遵守的反模式门：")
            for blocker in blockers:
                lines.append(
                    f"- {blocker['applicable_situation']} 禁止结果：{', '.join(blocker['blocks_outcomes'])}。"
                )
        else:
            lines.append("必须遵守的反模式门（若中止条件已成立，则忽略对应门）：")
            for blocker in blockers:
                operations = "；".join(blocker["operator_steps"])
                abort = "；".join(blocker["abort_conditions"])
                lines.append(
                    f"- {blocker['applicable_situation']} 禁止结果：{', '.join(blocker['blocks_outcomes'])}。"
                    f"检查动作：{operations}。中止条件：{abort}。"
                )
    card = decision.selected_card
    if card is None:
        lines.append("本集不注入策略卡；严格按分集目标、状态与硬锚点自主规划。")
        return "\n".join(lines)
    lines.extend([
        f"候选算子 [{card['mechanism_family']}]",
        f"适用情境：{card['applicable_situation']}",
    ])
    if include_steps:
        steps = "\n".join(f"  {index}. {step}" for index, step in enumerate(card["operator_steps"], start=1))
        lines.append(f"执行步骤：\n{steps}")
    lines.extend([
        f"预期变化：{card['expected_state_change']}",
        f"中止条件：{'；'.join(card['abort_conditions'])}",
        "若与任一硬锚点冲突，必须放弃该算子，不得改写硬锚点。",
    ])
    return "\n".join(lines)
