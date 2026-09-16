"""叙事义务记忆：六类义务的写入、去重与兑现生命周期。"""
from __future__ import annotations

import copy
from typing import Any

OBLIGATION_TYPES = {
    "FORESHADOW", "MYSTERY", "PROMISE",
    "PLAN", "DEADLINE", "RELATIONSHIP_DEBT",
}

# 结构层义务（去掉关系债）：关系债本质是人际张力，应交由人物关系记忆处理，
# 避免被义务记忆的「兑现/闭环」逻辑改写成可执行计划（RELATIONSHIP_PROCEDURALIZATION）。
STRUCTURAL_OBLIGATION_TYPES = {
    "FORESHADOW", "MYSTERY", "PROMISE", "PLAN", "DEADLINE",
}

# 单集兑现上限：防止模型在剧情高潮/关系转折集把大量未真正兑现的义务一次性标为 resolved。
MAX_RESOLVE_PER_EPISODE = 3


def _overlap(a: str, b: str) -> float:
    sa = {a[i:i + 2] for i in range(max(0, len(a) - 1))}
    sb = {b[i:i + 2] for i in range(max(0, len(b) - 1))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def validate_obligation_fields(result: dict[str, Any], obligation_types: set[str] | frozenset[str] = OBLIGATION_TYPES) -> None:
    new_items = result.get("new_obligations", [])
    resolved = result.get("resolved_obligations", [])
    if not isinstance(new_items, list) or not isinstance(resolved, list):
        raise ValueError("new_obligations / resolved_obligations 必须是数组")
    for index, item in enumerate(new_items):
        if not isinstance(item, dict):
            raise ValueError(f"new_obligations[{index}] 必须是对象")
        if item.get("type") not in obligation_types:
            raise ValueError(f"new_obligations[{index}].type 非法")
        for field in ("description", "required_payoff"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"new_obligations[{index}].{field} 必须是非空字符串")
        participants = item.get("participants", [])
        if not isinstance(participants, list) or any(not isinstance(x, str) for x in participants):
            raise ValueError(f"new_obligations[{index}].participants 必须是字符串数组")
        deadline = item.get("deadline")
        if deadline is not None and not isinstance(deadline, str):
            raise ValueError(f"new_obligations[{index}].deadline 必须是字符串或 null")
        if item.get("type") == "DEADLINE" and (not isinstance(deadline, str) or not deadline.strip()):
            raise ValueError(f"new_obligations[{index}].deadline 在 DEADLINE 类型下必须是非空字符串")
    if any(not isinstance(item, str) for item in resolved):
        raise ValueError("resolved_obligations 必须是字符串数组")


def apply_obligation_lifecycle(
    open_obligations: list[dict[str, Any]],
    *,
    new_obligations: list[dict[str, Any]],
    resolved_descriptions: list[str],
    job_id: str,
    episode_id: str,
    allowed_types: set[str] | frozenset[str] = OBLIGATION_TYPES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """兑现旧义务并加入本集新义务；返回 (open_after, added, resolved)。

    allowed_types 同时约束恢复的旧义务和本集新增义务，避免断点续跑时被旧版记录污染。
    """
    # 兜底：截断到单集上限，避免一次性清空开放义务。
    resolved_descriptions = list(resolved_descriptions)[:MAX_RESOLVE_PER_EPISODE]
    remaining: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []
    for obligation in open_obligations:
        if obligation.get("type") not in allowed_types:
            continue
        description = str(obligation.get("description", ""))
        matched = any(description == text or _overlap(description, text) > 0.5 for text in resolved_descriptions)
        if matched:
            item = copy.deepcopy(obligation)
            item["status"] = "RESOLVED"
            item["resolved_episode"] = episode_id
            resolved.append(item)
        else:
            remaining.append(copy.deepcopy(obligation))

    added: list[dict[str, Any]] = []
    known_descriptions = [str(item.get("description", "")) for item in remaining]
    for index, candidate in enumerate(new_obligations, start=1):
        if candidate.get("type") not in allowed_types:
            continue
        description = candidate["description"].strip()
        if any(description == old or _overlap(description, old) > 0.6 for old in known_descriptions):
            continue
        item = {
            "obligation_id": f"{job_id}__OB{index:02d}",
            "type": candidate["type"],
            "description": description,
            "participants": [x.strip() for x in candidate.get("participants", []) if x.strip()],
            "required_payoff": candidate["required_payoff"].strip(),
            "deadline": candidate.get("deadline"),
            "introduced_episode": episode_id,
            "resolved_episode": None,
            "status": "OPEN",
        }
        added.append(item)
        remaining.append(copy.deepcopy(item))
        known_descriptions.append(description)
    return remaining, added, resolved
