from __future__ import annotations

import copy
import json
from typing import Any

RELATIONSHIP_STATUSES = {"OPEN", "ACKNOWLEDGED", "RESPONDED", "TRANSFORMED", "RESOLVED"}
RELATIONSHIP_EVENTS = {"ACKNOWLEDGED", "RESPONDED", "TRANSFORMED", "RESOLVED"}
MAX_ACTIVE_RELATIONSHIPS = 12


def _bigrams(text: str) -> set[str]:
    clean = "".join(str(text).split())
    return {clean[index:index + 2] for index in range(max(0, len(clean) - 1))}


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def coerce_relationship_extraction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("人物关系记忆抽取结果必须是对象")
    normalized = copy.deepcopy(value)
    for wrapper in ("result", "output", "data"):
        nested = normalized.get(wrapper)
        if isinstance(nested, dict) and "new_relationship_memories" not in normalized:
            normalized = copy.deepcopy(nested)
            break
    if "new_relationship_memories" not in normalized or not isinstance(normalized["new_relationship_memories"], list):
        raise ValueError("new_relationship_memories 缺失或不是数组")
    if "relationship_updates" not in normalized or not isinstance(normalized["relationship_updates"], list):
        raise ValueError("relationship_updates 缺失或不是数组")
    return normalized


def validate_relationship_extraction(value: dict[str, Any], *, active_ids: set[str]) -> dict[str, Any]:
    new_memories = value.get("new_relationship_memories")
    updates = value.get("relationship_updates")
    if not isinstance(new_memories, list) or len(new_memories) > 1:
        raise ValueError("new_relationship_memories 必须是最多 1 条的数组")
    if not isinstance(updates, list):
        raise ValueError("relationship_updates 必须是数组")

    clean_new: list[dict[str, Any]] = []
    required_new = (
        "parties", "current_stance", "tension_source", "power_asymmetry", "debt_holder",
        "affected_party", "emotional_residue", "required_response", "unacceptable_shortcuts",
    )
    for index, item in enumerate(new_memories):
        if not isinstance(item, dict):
            raise ValueError(f"new_relationship_memories[{index}] 必须是对象")
        parties = item.get("parties")
        if not isinstance(parties, list) or len({str(x).strip() for x in parties if str(x).strip()}) < 2:
            raise ValueError(f"new_relationship_memories[{index}].parties 至少包含两人")
        normalized: dict[str, Any] = {
            "parties": list(dict.fromkeys(str(x).strip() for x in parties if str(x).strip())),
        }
        for field in required_new[1:]:
            raw = item.get(field)
            if field == "unacceptable_shortcuts":
                if not isinstance(raw, list) or not raw:
                    raise ValueError(f"new_relationship_memories[{index}].unacceptable_shortcuts 必须是非空数组")
                normalized[field] = [str(x).strip() for x in raw if str(x).strip()]
            else:
                text = str(raw or "").strip()
                if not text:
                    raise ValueError(f"new_relationship_memories[{index}].{field} 不能为空")
                normalized[field] = text
        evidence = str(item.get("evidence", "")).strip()
        if not evidence:
            # 容错：模型偶发不提供逐字引文，改用张力来源作可追溯线索，不终止轨迹。
            evidence = f"（无逐字引文）{normalized['tension_source'][:60]}"
        normalized["evidence"] = evidence
        clean_new.append(normalized)

    clean_updates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(updates):
        if not isinstance(item, dict):
            raise ValueError(f"relationship_updates[{index}] 必须是对象")
        relationship_id = str(item.get("relationship_id", "")).strip()
        event = str(item.get("event", "")).strip().upper()
        if relationship_id not in active_ids or relationship_id in seen:
            raise ValueError(f"relationship_updates[{index}].relationship_id 非法或重复")
        if event not in RELATIONSHIP_EVENTS:
            raise ValueError(f"relationship_updates[{index}].event 非法")
        evidence = str(item.get("evidence", "")).strip()
        if not evidence:
            raise ValueError(f"relationship_updates[{index}].evidence 不能为空")
        affected_response = str(item.get("affected_party_response", "")).strip()
        relationship_change = str(item.get("relationship_change", "")).strip()
        if event in {"RESPONDED", "RESOLVED"} and not affected_response:
            raise ValueError(f"relationship_updates[{index}] 的 {event} 必须有受影响方回应")
        if event in {"TRANSFORMED", "RESOLVED"} and not relationship_change:
            raise ValueError(f"relationship_updates[{index}] 的 {event} 必须有关系变化")
        clean_updates.append({
            "relationship_id": relationship_id,
            "event": event,
            "evidence": evidence,
            "affected_party_response": affected_response,
            "relationship_change": relationship_change,
            "current_stance": str(item.get("current_stance", "")).strip(),
            "emotional_residue": str(item.get("emotional_residue", "")).strip(),
        })
        seen.add(relationship_id)
    return {"new_relationship_memories": clean_new, "relationship_updates": clean_updates}


def apply_relationship_lifecycle(
    active_memories: list[dict[str, Any]],
    *,
    new_memories: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    job_id: str,
    episode_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    active = [copy.deepcopy(item) for item in active_memories]
    by_id = {item["relationship_id"]: item for item in active}
    changed: list[dict[str, Any]] = []
    resolved: list[dict[str, Any]] = []

    for update in updates:
        memory = by_id.get(update["relationship_id"])
        if memory is None:
            continue
        event = update["event"]
        memory["status"] = event
        if update.get("current_stance"):
            memory["current_stance"] = update["current_stance"]
        if update.get("emotional_residue"):
            memory["emotional_residue"] = update["emotional_residue"]
        memory["last_progress_episode"] = episode_id
        memory.setdefault("history", []).append({
            "episode_id": episode_id,
            "event": event,
            "evidence": update["evidence"],
            "affected_party_response": update.get("affected_party_response", ""),
            "relationship_change": update.get("relationship_change", ""),
        })
        if event == "RESOLVED":
            memory["resolved_episode"] = episode_id
            resolved.append(copy.deepcopy(memory))
        changed.append(copy.deepcopy(memory))

    active = [item for item in active if item.get("status") != "RESOLVED"]
    added: list[dict[str, Any]] = []
    for index, candidate in enumerate(new_memories, start=1):
        if len(active) >= MAX_ACTIVE_RELATIONSHIPS:
            break
        parties = set(candidate["parties"])
        duplicate = any(
            parties == set(item.get("parties", []))
            and _overlap(candidate["tension_source"], item.get("tension_source", "")) > 0.55
            for item in active
        )
        if duplicate:
            continue
        memory = {
            "relationship_id": f"REL_{job_id}_{index:02d}",
            **copy.deepcopy(candidate),
            "status": "OPEN",
            "source_episode": episode_id,
            "last_progress_episode": episode_id,
            "resolved_episode": None,
            "history": [],
        }
        active.append(memory)
        added.append(copy.deepcopy(memory))
    return active, added, changed, resolved


def looks_like_relationship_plan(candidate: dict[str, Any], relationship_memories: list[dict[str, Any]]) -> bool:
    """识别把关系债改写成 PLAN 的候选，供双层模式确定性分流。"""
    if candidate.get("type") != "PLAN":
        return False
    text = f"{candidate.get('description', '')} {candidate.get('required_payoff', '')}"
    markers = ("解释", "道歉", "原谅", "和解", "信任", "坦白", "说清", "隐瞒", "补偿", "关系", "误会")
    if not any(marker in text for marker in markers):
        return False
    candidate_parties = {str(item).strip() for item in candidate.get("participants", []) if str(item).strip()}
    for memory in relationship_memories:
        memory_parties = set(memory.get("parties", []))
        if candidate_parties and not (candidate_parties & memory_parties):
            continue
        relation_text = f"{memory.get('tension_source', '')} {memory.get('required_response', '')}"
        if _overlap(text, relation_text) > 0.12 or any(marker in relation_text for marker in markers):
            return True
    return False


def generation_view(active_memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "relationship_id", "parties", "current_stance", "tension_source", "power_asymmetry",
        "debt_holder", "affected_party", "emotional_residue", "required_response",
        "unacceptable_shortcuts", "status", "source_episode", "last_progress_episode",
    )
    return [{field: item.get(field) for field in fields} for item in active_memories]


def build_relationship_extraction_prompt(
    *,
    story: dict[str, Any],
    episode_id: str,
    script: str,
    active_memories: list[dict[str, Any]],
) -> str:
    characters = [{"name": item["name"], "role": item["role"]} for item in story["characters"]]
    return f"""你是连续剧人物关系记忆标注器。只根据当前集剧本识别真正需要跨集保留的人际债务，并更新已有关系记忆；不修复、不补写、不评价文笔。只输出合法 JSON，不要 Markdown。

故事：{story['title']}
人物：{json.dumps(characters, ensure_ascii=False)}
当前集：{episode_id}
当前开放人物关系记忆：
{json.dumps(generation_view(active_memories), ensure_ascii=False)}

当前集剧本：
{script}

输出格式：
{{
  "new_relationship_memories": [{{
    "parties": ["至少两名人物"],
    "current_stance": "双方当前立场与相处边界",
    "tension_source": "本集新产生的具体伤害、亏欠、隐瞒、救助或不对等交换",
    "power_asymmetry": "谁掌握信息/资源/决定权，以及不对等在哪里；若对等则写对等",
    "debt_holder": "负有回应责任的一方",
    "affected_party": "有权接受、拒绝或提出条件的一方",
    "emotional_residue": "本集结束后仍会影响后续行为的情绪残留",
    "required_response": "未来怎样的双方回应或关系变化才算真正结清",
    "unacceptable_shortcuts": ["完成一个计划不算结清", "单方面解释或道歉不算结清"],
    "evidence": "当前集剧本中的具体证据"
  }}],
  "relationship_updates": [{{
    "relationship_id": "当前开放记忆中的原 ID",
    "event": "ACKNOWLEDGED|RESPONDED|TRANSFORMED|RESOLVED",
    "evidence": "当前集具体动作或台词",
    "affected_party_response": "受影响方可观察的接受/拒绝/条件/选择；没有则空字符串",
    "relationship_change": "信任、权力、边界或后续选择的实际变化；没有则空字符串",
    "current_stance": "更新后的双方立场；无变化则空字符串",
    "emotional_residue": "更新后仍未消失的情绪残留；无变化则空字符串"
  }}]
}}

严格规则：
1. new_relationship_memories 每集默认 0 条，最多 1 条。只有【明确伤害、亏欠、隐瞒、救助或交换】造成未来必须回应的人际债务才记录。普通争吵、意见不同、合作分工、调查计划、轻微情绪波动不记录。
2. 同一关系债已在开放记忆中时不得重复新建；只在本集有可观察变化时输出 relationship_updates，未触及则不输出。
3. ACKNOWLEDGED：债务被人物明确意识到或承认，但尚无受影响方回应。
4. RESPONDED：受影响方已经以接受、拒绝、提出条件、保留信任或改变选择作出可观察回应。只有负债方完成计划、解释、道歉或谈话，不算 RESPONDED。
5. TRANSFORMED：双方的信任、权力、关系边界或后续选择已经实际改变，但债务可能仍未完全结清；必须写 relationship_change。
6. RESOLVED：必须同时满足【受影响方有可观察回应】和【关系的信任/权力/边界/后续选择实际改变】。完成任务、执行计划、见面谈话、单方面解释或道歉、剧情进入高潮，都不能单独结清关系债。
7. 人物可以拒绝、延迟、谈条件或不原谅。不要为了让剧情顺利而把受影响方写成自动配合。
8. required_response 必须是人物关系层的回应，不得写成「完成某任务/调查/比赛/提交文件」。unacceptable_shortcuts 至少列两条本债务最容易被错误简化的捷径。
9. 证据必须来自当前集剧本；拿不准时宁可不新增、不升级。字符串内引用原话用中文引号「」，禁止英文单引号。
10. 最外层只输出上述对象，不要包在 result/output/data 字段。"""
