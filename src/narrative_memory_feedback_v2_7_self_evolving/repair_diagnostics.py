"""V4 repair 卡的零 API 诊断器。

诊断发生在生成当前集之前，只使用已经生成的剧本、结构化 extraction、当前生命周期状态
和本集大纲。规则刻意保守：宁可少触发，也避免把正常叙事误判为故障。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .narrative_signature import _dialogue_similarity, _extract_boundary_dialogue
from .retrieval_v4 import infer_episode_function


@dataclass
class RepairDiagnosticResult:
    tags: set[str] = field(default_factory=set)
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, tag: str, **evidence: Any) -> None:
        self.tags.add(tag)
        self.evidence[tag] = evidence


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _cjk_len(text: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", text))


def _boundary_meta(script: str, *, tail: bool) -> tuple[str, set[str]]:
    region = script[-1200:] if tail else script[:1200]
    locations = re.findall(r"(?:\*{0,2})地点[：:](?:\*{0,2})\s*([^\n]+)", region)
    location = (locations[-1] if tail and locations else locations[0] if locations else "").strip(" *#")
    names = set(re.findall(r"\*\*([\u4e00-\u9fff]{2,4})\*\*", region))
    names.update(re.findall(r"(?:^|\n)([\u4e00-\u9fff]{2,4})(?:[（(]|[：:])", region))
    names = {
        name for name in names
        if name not in {"时间", "地点", "人物", "场景", "动作", "内景", "外景"}
        and not name.startswith(("场景", "第几"))
    }
    return location, names


def _location_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a = re.sub(r"[的\s*#]", "", a)
    b = re.sub(r"[的\s*#]", "", b)
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.8
    aa = {a[i:i + 2] for i in range(len(a) - 1)}
    bb = {b[i:i + 2] for i in range(len(b) - 1)}
    return len(aa & bb) / max(len(aa | bb), 1) if aa and bb else 0.0


def _robust_boundary_overlap(previous_script: str, latest_script: str) -> dict[str, Any]:
    tail_location, tail_chars = _boundary_meta(previous_script, tail=True)
    head_location, head_chars = _boundary_meta(latest_script, tail=False)
    location_overlap = _location_similarity(tail_location, head_location)
    character_overlap = (
        len(tail_chars & head_chars) / len(tail_chars | head_chars)
        if tail_chars and head_chars else 0.0
    )
    tail_dialogue = _extract_boundary_dialogue(previous_script, is_tail=True)
    head_dialogue = _extract_boundary_dialogue(latest_script, is_tail=False)
    dialogue_similarity = _dialogue_similarity(tail_dialogue, head_dialogue)
    dialogue_norm = min(1.0, dialogue_similarity / 0.10)
    score = 0.35 * location_overlap + 0.30 * character_overlap + 0.35 * dialogue_norm
    return {
        "boundary_overlap_score": round(score, 4),
        "location_overlap": round(location_overlap, 4),
        "character_overlap": round(character_overlap, 4),
        "dialogue_similarity": round(dialogue_similarity, 4),
        "tail_location": tail_location,
        "head_location": head_location,
        "shared_characters": sorted(tail_chars & head_chars),
    }


def _dialogue_and_action_stats(script: str) -> tuple[int, int, int, int]:
    dialogue_chars = 0
    action_chars = 0
    long_dialogue_lines = 0
    action_lines = 0
    dialogue_pattern = re.compile(
        r"^(?:\*{0,2})?[\u4e00-\u9fff]{2,4}(?:\*{0,2})?"
        r"(?:[（(][^）)]*[）)])?[：:]|^——"
    )
    bold_name_pattern = re.compile(r"^\*\*[\u4e00-\u9fff]{2,4}\*\*$")
    waiting_dialogue = False
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            continue
        if bold_name_pattern.match(line):
            waiting_dialogue = True
            continue
        if waiting_dialogue and line.startswith(("（", "(")):
            continue
        n = _cjk_len(line)
        if waiting_dialogue or dialogue_pattern.search(line):
            dialogue_chars += n
            if n >= 55:
                long_dialogue_lines += 1
            waiting_dialogue = False
        elif line.startswith(("△", "【", "（", "(", "【动作】", "动作：")):
            action_chars += n
            action_lines += 1
        elif not line.startswith(("#", "*", "---")) and not re.match(r"^(?:时间|地点|人物)[：:]", line):
            # 剧本中的无角色前缀叙述行也属于可视动作/场景描述。
            action_chars += n
            action_lines += 1
    return dialogue_chars, action_chars, long_dialogue_lines, action_lines


def _detect_exposition(script: str) -> dict[str, Any] | None:
    if not script:
        return None
    dialogue_chars, action_chars, long_lines, action_lines = _dialogue_and_action_stats(script)
    total = dialogue_chars + action_chars
    if total < 500:
        return None
    markers = [
        "也就是说", "换句话说", "你应该知道", "你也知道", "我们都知道",
        "这意味着", "事实上", "问题在于", "所以说", "总而言之",
        "说到底", "简单来说", "原因就是", "这说明", "归根结底",
    ]
    marker_count = sum(script.count(marker) for marker in markers)
    dialogue_ratio = dialogue_chars / max(total, 1)
    # 台词占比和长台词同时位于高尾部才触发；解释标记仅作为证据，不单独触发。
    exposition_heavy = (
        (dialogue_ratio >= 0.60 and long_lines >= 3)
        or (dialogue_ratio >= 0.67 and long_lines >= 2)
    )
    if exposition_heavy:
        return {
            "dialogue_ratio": round(dialogue_ratio, 3),
            "exposition_marker_count": marker_count,
            "long_dialogue_lines": long_lines,
            "action_lines": action_lines,
        }
    return None


def _detect_causal_gap(previous_extraction: dict[str, Any] | None) -> dict[str, Any] | None:
    if not previous_extraction:
        return None
    issues = _as_list(previous_extraction.get("state_consistency_issues"))
    negative_issues = [
        str(item) for item in issues
        if not any(ok in str(item) for ok in ("符合", "未违反", "未出现", "无冲突", "未越界"))
    ]
    issue_text = " ".join(negative_issues)
    causal_terms = ("因果断裂", "无依据", "缺少依据", "缺乏依据", "逻辑跳跃", "未交代", "无法解释", "没有支撑")
    matched_terms = sorted({term for term in causal_terms if term in issue_text})
    if matched_terms:
        return {"source": "state_consistency_issues", "matched_terms": matched_terms, "issues": negative_issues[:3]}

    # 保守兜底：上一集一次性关闭多个目标/未知，但没有写入任何事实、证据或时间线支撑。
    resolved_count = len(_as_list(previous_extraction.get("resolved_goals"))) + len(_as_list(previous_extraction.get("resolved_unknown")))
    delta = previous_extraction.get("state_delta", {})
    support_count = sum(len(_as_list(delta.get(field))) for field in (
        "confirmed_facts", "resources_and_evidence", "known_information", "timeline",
    )) if isinstance(delta, dict) else 0
    if resolved_count >= 2 and support_count == 0:
        return {"source": "unsupported_multi_resolution", "resolved_count": resolved_count, "support_count": 0}
    return None


def _detect_anchor_conflict(job: dict[str, Any], plan: dict[str, Any], total_episodes: int) -> dict[str, Any] | None:
    anchors = [str(x) for x in _as_list(plan.get("hard_anchors")) if str(x).strip()]
    if not anchors:
        return None
    protected = [a for a in anchors if any(term in a for term in (
        "不得确认", "不能确认", "不可确认", "不允许确认", "不揭示", "不得揭示", "不能揭示",
        "不得锁定", "不能锁定", "不得证明", "不能证明", "保持未知", "仍未揭晓", "尚未揭晓",
        "不得彻底和解", "不能彻底和解", "不得恢复关系", "避免锁定",
    ))]
    if not protected:
        return None
    episode_function, _ = infer_episode_function(job, total_episodes)
    if episode_function not in {"TEST", "PRESERVE", "CONSOLIDATE", "CLOSE"}:
        return None
    target_text = " ".join([
        str(plan.get("episode_goal", "")),
        *[str(x) for x in _as_list(plan.get("open_decisions"))],
    ])
    lock_terms = sorted({term for term in (
        "确认", "揭示", "查明", "锁定", "证明", "认定", "揭晓", "公开真相", "最终决定", "彻底解决", "和解", "恢复关系",
    ) if term in target_text})
    if lock_terms:
        return {
            "episode_function": episode_function,
            "lock_terms": lock_terms,
            "protected_anchors": protected[:3],
        }
    return None


def _detect_closure_overload(job: dict[str, Any], state: dict[str, Any], total_episodes: int) -> dict[str, Any] | None:
    episode_function, _ = infer_episode_function(job, total_episodes)
    # 只在最后两集触发。CONSOLIDATE 关键词过宽，不能等同于真正收束窗口。
    if episode_function != "CLOSE" or int(job.get("episode_index", 0)) < max(total_episodes - 2, 0):
        return None
    counts = {
        field: len(_as_list(state.get(field)))
        for field in ("current_goals", "unknown_information", "unresolved_threads")
    }
    active_categories = sum(1 for count in counts.values() if count >= 2)
    total_open = sum(counts.values())
    # 最终窗口仍有三类开放债务且总数不少于 10，才判为收束过载。
    if active_categories >= 3 and total_open >= 10:
        return {"episode_function": episode_function, "open_counts": counts, "total_open": total_open}
    return None


def detect_repair_diagnostics(
    *,
    job: dict[str, Any],
    plan: dict[str, Any],
    state: dict[str, Any],
    recent_scripts: list[str],
    previous_extraction: dict[str, Any] | None,
    total_episodes: int,
) -> RepairDiagnosticResult:
    """生成当前集前，检测 5 类尚未接通的 repair 诊断信号。"""
    result = RepairDiagnosticResult()

    # 前两集的边界已出现高重叠，则让当前集主动换场景功能；阈值沿用 surface 高重复定义。
    if len(recent_scripts) >= 2:
        overlap = _robust_boundary_overlap(recent_scripts[-2], recent_scripts[-1])
        if (
            float(overlap.get("boundary_overlap_score", 0.0)) > 0.70
            and bool(overlap.get("tail_location"))
            and bool(overlap.get("head_location"))
        ):
            result.add("recent_scene_repeat", **overlap)

    exposition = _detect_exposition(recent_scripts[-1] if recent_scripts else "")
    if exposition:
        result.add("exposition_heavy", **exposition)

    causal_gap = _detect_causal_gap(previous_extraction)
    if causal_gap:
        result.add("causal_gap", **causal_gap)

    anchor_conflict = _detect_anchor_conflict(job, plan, total_episodes)
    if anchor_conflict:
        result.add("anchor_conflict_risk", **anchor_conflict)

    closure_overload = _detect_closure_overload(job, state, total_episodes)
    if closure_overload:
        result.add("closure_overload", **closure_overload)

    return result
