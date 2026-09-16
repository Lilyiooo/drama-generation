"""受控状态标签提取器（V4 关键词版，供 V4 门控经验卡检索使用）。

合并两段逻辑：
1. V3 的 extract_state_tags —— 变化敏感的状态标签 + 最近 N 集 Move 分布退行检测。
2. V4 的 _add_orphan_tags —— 补全卡池 requires 引用但 V3 未生成的 9 个孤儿 tag。

不依赖 LLM（语义标签 annotate_state_tags 在归因实验中证明不是关键变量，且额外 API 成本）。

返回 (tags, trajectory_declining)。
"""
from __future__ import annotations

from typing import Any


def extract_state_tags(
    job: dict[str, Any],
    state: dict[str, Any],
    recent_moves: list[str],
    previous_state: dict[str, Any] | None = None,
) -> tuple[set[str], bool]:
    """从结构化 state + job + 近期 move + 前态提取受控标签。

    变化敏感原则：状态标签应反映「本集之间发生了什么变化/停滞」，而非「字段是否非空」，
    避免长期恒真的常驻标签稀释检索信号。
    """
    tags: set[str] = set()
    prev = previous_state if isinstance(previous_state, dict) else {}

    unknown_info = state.get("unknown_information", [])
    known_info = state.get("known_information", [])
    confirmed = state.get("confirmed_facts", [])
    resources = state.get("resources_and_evidence", [])
    goals = state.get("current_goals", [])
    unresolved = state.get("unresolved_threads", [])
    relationship = state.get("relationship_state", {})
    timeline = state.get("timeline", [])
    character_state = state.get("character_state", {})

    p_known = prev.get("known_information", [])
    p_confirmed = prev.get("confirmed_facts", [])
    p_resources = prev.get("resources_and_evidence", [])
    p_goals = prev.get("current_goals", [])
    p_unresolved = prev.get("unresolved_threads", [])
    p_relationship = prev.get("relationship_state", {})
    p_character = prev.get("character_state", {})

    def _n(x: Any) -> int:
        return len(x) if isinstance(x, (list, dict)) else 0

    has_prev = bool(prev)

    def grew(cur: Any, old: Any) -> bool:
        """本集之间该字段有新增（追加式合并下 len 单调不减）。"""
        return has_prev and _n(cur) > _n(old)

    def stagnant(cur: Any, old: Any) -> bool:
        """本集之间该字段非空但无新增 → 停滞。"""
        return has_prev and _n(cur) > 0 and _n(cur) == _n(old)

    # ── trajectory detection: 基于最近 N 集 Move 走向判断退行（可翻转，非累积）──
    DECLINING_MOVES = {
        "INFO_LOSS", "INFO_REFRAME", "OPTION_LOSS", "OPTION_REFRAME",
        "BOND_LOSS", "BOND_REFRAME", "CAPACITY_LOSS", "CAPACITY_REFRAME",
    }
    GAIN_MOVES = {"INFO_GAIN", "OPTION_GAIN", "BOND_GAIN", "CAPACITY_GAIN"}
    DECLINING_WINDOW = 4
    recent_mv = [m for m in recent_moves[-DECLINING_WINDOW:] if m]
    loss_cnt = sum(1 for m in recent_mv if m in DECLINING_MOVES)
    gain_cnt = sum(1 for m in recent_mv if m in GAIN_MOVES)
    trajectory_declining = bool(recent_mv) and loss_cnt >= 2 and loss_cnt > gain_cnt

    rel_texts = [s for v in relationship.values() for s in (v if isinstance(v, list) else [v])]
    char_texts = [s for v in character_state.values() for s in (v if isinstance(v, list) else [v])]

    # ── information ──
    if unknown_info and not grew(known_info, p_known) and not grew(confirmed, p_confirmed):
        tags.add("actionable_unknown")
    if grew(resources, p_resources) and _n(resources) >= 2:
        tags.add("multiple_information_sources")
    if unknown_info and resources and not grew(confirmed, p_confirmed):
        tags.add("action_can_test_belief")
    if any("期限" in t or "截止" in t or "窗口" in t or "限期" in t for t in timeline):
        tags.add("access_window_limited")
    if _n(unknown_info) >= 2 and not any("锁定" in t or "确认" in t for t in confirmed):
        tags.add("competing_explanations")
    if grew(known_info, p_known) and _n(known_info) >= 2 and not grew(confirmed, p_confirmed):
        tags.add("existing_fact_reinterpretable")
    if unknown_info and resources and stagnant(confirmed, p_confirmed):
        tags.add("verification_opportunity")
    if _n(unknown_info) >= 2 and not grew(confirmed, p_confirmed):
        tags.add("answer_not_yet_earned")

    # ── option ──
    if goals and stagnant(goals, p_goals):
        tags.add("active_goal_blocked")
    if grew(resources, p_resources):
        tags.add("exchangeable_resource")
    if grew(unresolved, p_unresolved):
        tags.add("alternative_route_possible")
    if any(d in t for d in ("期限", "截止", "到期", "之前") for field in (timeline + goals) for t in [str(field)]):
        tags.add("active_deadline")
    if _n(resources) <= 2 and _n(goals) >= 2:
        tags.add("resource_scarcity")
    if resources and stagnant(resources, p_resources):
        tags.add("reusable_existing_resource")
    if any("先" in t or "之后" in t or "前提" in t for t in goals):
        tags.add("sequence_bottleneck")
    if _n(unresolved) >= 2 and grew(unresolved, p_unresolved):
        tags.add("multiple_viable_options")

    # ── bond ──
    if any("合作" in s or "共同" in s or "协作" in s for s in goals):
        tags.add("shared_task_exists")
    if any("信任不足" in s or "冷战" in s or "对峙" in s for s in rel_texts):
        tags.add("trust_low")
    if any("信赖" in s or "验证" in s or "合作" in s for s in rel_texts):
        tags.add("reliability_test_available")
    if any("冲突" in s or "边界" in s or "权限" in s for s in rel_texts) and grew(relationship, p_relationship):
        tags.add("relationship_boundary_active")
    if any("承担" in s or "负责" in s for s in char_texts) and grew(character_state, p_character):
        tags.add("unequal_cost_distribution")
    if any("责任" in s or "分工" in s or "谁" in s for s in rel_texts):
        tags.add("responsibility_dispute")
    if grew(relationship, p_relationship):
        tags.add("cooperation_ready_to_formalize")

    # ── capacity ──
    if any("提升" in t or "训练" in t or "工序" in t for t in goals):
        tags.add("incremental_skill_task")
    if grew(goals, p_goals) and _n(goals) >= 2:
        tags.add("transferable_subtask")
    last_move = recent_moves[-1] if recent_moves else ""
    if last_move and ("LOSS" in last_move or "REFRAME" in last_move):
        tags.add("prior_attempt_failed")
    if any("接近" in t or "承受" in t or "负荷" in t for t in goals):
        tags.add("capacity_under_strain")
    if _n(resources) >= 2 and _n(goals) >= 2:
        tags.add("complementary_capabilities")

    # ── trajectory-aware signals ──
    if trajectory_declining:
        tags.add("vulnerable_information_source")
        tags.add("capacity_under_strain")
        tags.add("resource_scarcity")
        tags.discard("complementary_capabilities")
        tags.discard("shared_task_exists")

    # ── boundary: 约束与边界信号 ──
    if confirmed and not unresolved:
        tags.add("core_problem_already_resolved")
    if job.get("hard_anchors"):
        tags.add("anchor_forbids_lock")
    if goals and not unresolved and not resources:
        tags.add("no_real_alternative")
    if _n(confirmed) >= 2 and not unresolved:
        tags.add("irreversible_choice_ready")

    # ── repair diagnostics ──
    MOVE_REPEAT_COUNT = 3
    if len(recent_moves) >= MOVE_REPEAT_COUNT and len(set(recent_moves[-MOVE_REPEAT_COUNT:])) <= 2:
        tags.add("recent_move_repeat")
    if has_prev and all(not grew(c, o) for c, o in (
        (known_info, p_known), (confirmed, p_confirmed), (resources, p_resources),
        (goals, p_goals), (unresolved, p_unresolved), (relationship, p_relationship),
    )):
        tags.add("state_stagnation")

    _add_orphan_tags(job, state, tags, previous_state, grew)
    return tags, trajectory_declining


def _add_orphan_tags(
    job: dict[str, Any],
    state: dict[str, Any],
    tags: set[str],
    previous_state: dict[str, Any] | None,
    grew,
) -> None:
    """补全 V3 未生成但卡池 requires 引用的 9 个孤儿 tag（关键词判定）。"""
    prev = previous_state if isinstance(previous_state, dict) else {}
    has_prev = bool(prev)

    def _n(x: Any) -> int:
        return len(x) if isinstance(x, (list, dict)) else 0

    confirmed = state.get("confirmed_facts", [])
    goals = state.get("current_goals", [])
    known_info = state.get("known_information", [])
    relationship = state.get("relationship_state", {})
    character_state = state.get("character_state", {})

    p_character = prev.get("character_state", {})

    rel_texts = [s for v in relationship.values() for s in (v if isinstance(v, list) else [v])]
    char_texts = [s for v in character_state.values() for s in (v if isinstance(v, list) else [v])]

    # sufficient_evidence_chain：证据链充分
    if _n(confirmed) >= 3 and job.get("episode_function") in {"CONSOLIDATE", "CLOSE"}:
        tags.add("sufficient_evidence_chain")

    # handoff_ready：关系或角色状态出现交接/移交/托付语义
    if any("交接" in s or "移交" in s or "托付" in s or "接手" in s for s in rel_texts + char_texts):
        tags.add("handoff_ready")

    # competence_demonstrated：角色状态出现胜任/证明/独当一面，且本集有新增
    if any("胜任" in s or "证明" in s or "独当一面" in s for s in char_texts) and grew(character_state, p_character):
        tags.add("competence_demonstrated")

    # agency_unavailable：核心人物能动性受限/缺席/无力
    if any("无力" in s or "缺席" in s or "无法" in s or "抽身" in s or "退出" in s for s in char_texts):
        tags.add("agency_unavailable")

    # relationship_basis_unstable：关系基础动摇/破裂/疏远
    if any("动摇" in s or "破裂" in s or "疏远" in s or "崩塌" in s for s in rel_texts):
        tags.add("relationship_basis_unstable")

    # role_recognition_pending：目标涉及认可/名分/晋升且尚未达成
    if any("认可" in s or "承认" in s or "晋升" in s or "名分" in s for s in goals):
        tags.add("role_recognition_pending")

    # diagnosed_method_failure：已有失败尝试，且已知信息记录了失败原因/教训
    if "prior_attempt_failed" in tags and any("失败" in s or "教训" in s or "原因" in s or "行不通" in s for s in known_info):
        tags.add("diagnosed_method_failure")

    # external_rescue_risk：目标出现求助/外援/依赖外部的信号
    if any("求助" in s or "外援" in s or "外部" in s or "搬救兵" in s for s in goals):
        tags.add("external_rescue_risk")
