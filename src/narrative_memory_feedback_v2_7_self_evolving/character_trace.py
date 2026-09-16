"""人物因果痕迹记忆：情绪痕迹 + 主体性痕迹。

只记录已经发生的触发、人物解释、选择、代价与可观察后果，不记录未来应做什么。
区别于 S4 relationship_memory：不写 required_response / unacceptable_shortcuts，
避免把人物记忆重新变成指令清单。
"""
from __future__ import annotations

import copy
import json
from typing import Any

EMOTIONAL_STATUSES = {"ACTIVE", "TRANSFORMED", "SETTLED"}
AGENCY_STATUSES = {"ACTIVE", "REVISED", "PAID", "SETTLED"}
EMOTIONAL_EVENTS = {"TRANSFORMED", "SETTLED"}
AGENCY_EVENTS = {"REVISED", "PAID", "SETTLED"}

MAX_EMOTIONAL_PER_CHAR = 2
MAX_AGENCY_PER_CHAR = 2
MAX_TOTAL_TRACES = 8

EMOTIONAL_FIELDS = (
    "trace_id", "character", "trigger_event", "appraisal", "residue",
    "unresolved_stake", "last_observed_effect", "source_episode",
    "last_update_episode", "status", "evidence",
)
AGENCY_FIELDS = (
    "trace_id", "character", "decision", "motivation_source",
    "rejected_alternative", "accepted_cost", "pending_consequence",
    "last_observed_effect", "source_episode", "last_update_episode",
    "status", "evidence",
)


def coerce_character_trace_extraction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("人物因果痕迹抽取结果必须是对象")
    normalized = copy.deepcopy(value)
    for wrapper in ("result", "output", "data"):
        nested = normalized.get(wrapper)
        if isinstance(nested, dict) and "new_emotional_traces" not in normalized:
            normalized = copy.deepcopy(nested)
            break
    for key in ("new_emotional_traces", "emotional_updates", "new_agency_traces", "agency_updates"):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    return normalized


def _clean_emotional(item: dict[str, Any]) -> dict[str, Any]:
    character = str(item.get("character", "")).strip()
    if not character:
        raise ValueError("emotional_trace.character 不能为空")
    normalized: dict[str, Any] = {"character": character}
    for field in ("trigger_event", "appraisal", "residue", "unresolved_stake"):
        text = str(item.get(field, "")).strip()
        if not text:
            raise ValueError(f"emotional_trace.{field} 不能为空")
        normalized[field] = text
    normalized["last_observed_effect"] = str(item.get("last_observed_effect", "")).strip()
    evidence = str(item.get("evidence", "")).strip()
    if not evidence:
        evidence = f"（无逐字引文）{normalized['trigger_event'][:60]}"
    normalized["evidence"] = evidence
    return normalized


def _clean_agency(item: dict[str, Any]) -> dict[str, Any]:
    character = str(item.get("character", "")).strip()
    if not character:
        raise ValueError("agency_trace.character 不能为空")
    normalized: dict[str, Any] = {"character": character}
    for field in ("decision", "motivation_source", "rejected_alternative", "accepted_cost"):
        text = str(item.get(field, "")).strip()
        if not text:
            raise ValueError(f"agency_trace.{field} 不能为空")
        normalized[field] = text
    normalized["pending_consequence"] = str(item.get("pending_consequence", "")).strip()
    normalized["last_observed_effect"] = str(item.get("last_observed_effect", "")).strip()
    evidence = str(item.get("evidence", "")).strip()
    if not evidence:
        evidence = f"（无逐字引文）{normalized['decision'][:60]}"
    normalized["evidence"] = evidence
    return normalized


def validate_character_trace_extraction(
    value: dict[str, Any],
    *,
    active_emotional_ids: set[str],
    active_agency_ids: set[str],
) -> dict[str, Any]:
    new_emotional = value.get("new_emotional_traces", [])
    new_agency = value.get("new_agency_traces", [])
    emo_updates = value.get("emotional_updates", [])
    agy_updates = value.get("agency_updates", [])

    clean_emotional: list[dict[str, Any]] = []
    for item in new_emotional:
        if not isinstance(item, dict):
            raise ValueError("new_emotional_traces 元素必须是对象")
        clean_emotional.append(_clean_emotional(item))
    clean_agency: list[dict[str, Any]] = []
    for item in new_agency:
        if not isinstance(item, dict):
            raise ValueError("new_agency_traces 元素必须是对象")
        clean_agency.append(_clean_agency(item))

    def _clean_updates(updates: Any, active_ids: set[str], events: set[str]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in updates:
            if not isinstance(item, dict):
                raise ValueError("updates 元素必须是对象")
            trace_id = str(item.get("trace_id", "")).strip()
            event = str(item.get("event", "")).strip().upper()
            if trace_id not in active_ids or trace_id in seen:
                raise ValueError(f"update.trace_id 非法或重复：{trace_id}")
            if event not in events:
                raise ValueError(f"update.event 非法：{event}")
            evidence = str(item.get("evidence", "")).strip()
            if not evidence:
                raise ValueError("update.evidence 不能为空")
            cleaned.append({
                "trace_id": trace_id,
                "event": event,
                "evidence": evidence,
            })
            seen.add(trace_id)
        return cleaned

    return {
        "new_emotional_traces": clean_emotional,
        "emotional_updates": _clean_updates(emo_updates, active_emotional_ids, EMOTIONAL_EVENTS),
        "new_agency_traces": clean_agency,
        "agency_updates": _clean_updates(agy_updates, active_agency_ids, AGENCY_EVENTS),
    }


def apply_character_trace_lifecycle(
    emotional: list[dict[str, Any]],
    agency: list[dict[str, Any]],
    *,
    new_emotional: list[dict[str, Any]],
    emotional_updates: list[dict[str, Any]],
    new_agency: list[dict[str, Any]],
    agency_updates: list[dict[str, Any]],
    job_id: str,
    episode_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """返回 (emotional, agency, record)。"""
    emo = [copy.deepcopy(item) for item in emotional]
    agy = [copy.deepcopy(item) for item in agency]

    emo_by_id = {item["trace_id"]: item for item in emo}
    agy_by_id = {item["trace_id"]: item for item in agy}
    changed: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []

    for update in emotional_updates:
        trace = emo_by_id.get(update["trace_id"])
        if trace is None:
            continue
        trace["status"] = update["event"]
        trace["last_update_episode"] = episode_id
        changed.append(copy.deepcopy(trace))
        if update["event"] == "SETTLED":
            settled.append(copy.deepcopy(trace))
    for update in agency_updates:
        trace = agy_by_id.get(update["trace_id"])
        if trace is None:
            continue
        trace["status"] = update["event"]
        trace["last_update_episode"] = episode_id
        changed.append(copy.deepcopy(trace))
        if update["event"] == "SETTLED":
            settled.append(copy.deepcopy(trace))

    emo = [item for item in emo if item.get("status") != "SETTLED"]
    agy = [item for item in agy if item.get("status") != "SETTLED"]

    added: list[dict[str, Any]] = []
    for index, candidate in enumerate(new_emotional, start=1):
        if len(emo) + len(agy) >= MAX_TOTAL_TRACES:
            break
        if _char_count(emo, candidate["character"]) >= MAX_EMOTIONAL_PER_CHAR:
            continue
        trace = {
            "trace_id": f"EMO_{job_id}_{index:02d}",
            **copy.deepcopy(candidate),
            "status": "ACTIVE",
            "source_episode": episode_id,
            "last_update_episode": episode_id,
        }
        emo.append(trace)
        added.append(copy.deepcopy(trace))
    for index, candidate in enumerate(new_agency, start=1):
        if len(emo) + len(agy) >= MAX_TOTAL_TRACES:
            break
        if _char_count(agy, candidate["character"]) >= MAX_AGENCY_PER_CHAR:
            continue
        trace = {
            "trace_id": f"AGY_{job_id}_{index:02d}",
            **copy.deepcopy(candidate),
            "status": "ACTIVE",
            "source_episode": episode_id,
            "last_update_episode": episode_id,
        }
        agy.append(trace)
        added.append(copy.deepcopy(trace))

    record = {
        "added": added,
        "changed": changed,
        "settled": settled,
        "active_emotional_count": len(emo),
        "active_agency_count": len(agy),
    }
    return emo, agy, record


def _char_count(traces: list[dict[str, Any]], character: str) -> int:
    return sum(1 for item in traces if item.get("character") == character)


def generation_view(emotional: list[dict[str, Any]], agency: list[dict[str, Any]]) -> dict[str, Any]:
    def _emo(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "character": item.get("character", ""),
            "trigger_event": item.get("trigger_event", ""),
            "appraisal": item.get("appraisal", ""),
            "residue": item.get("residue", ""),
            "unresolved_stake": item.get("unresolved_stake", ""),
        }

    def _agy(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "character": item.get("character", ""),
            "decision": item.get("decision", ""),
            "motivation_source": item.get("motivation_source", ""),
            "rejected_alternative": item.get("rejected_alternative", ""),
            "accepted_cost": item.get("accepted_cost", ""),
            "pending_consequence": item.get("pending_consequence", ""),
        }

    return {
        "emotional_traces": [_emo(item) for item in emotional],
        "agency_traces": [_agy(item) for item in agency],
    }


def extraction_view(emotional: list[dict[str, Any]], agency: list[dict[str, Any]]) -> dict[str, Any]:
    """抽取标注器的活跃痕迹视图：必须暴露 trace_id，供 updates 逐字引用。

    与 generation_view 的区别：generation_view 只给生成模型看人物处境（不含 trace_id，
    避免生成 prompt 里出现无意义的内部 ID）；抽取模型需要精确引用上一条痕迹做
    TRANSFORMED/SETTLED，因此这里额外输出 trace_id 与 status。
    """
    def _emo(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "trace_id": item.get("trace_id", ""),
            "character": item.get("character", ""),
            "trigger_event": item.get("trigger_event", ""),
            "residue": item.get("residue", ""),
            "status": item.get("status", ""),
        }

    def _agy(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "trace_id": item.get("trace_id", ""),
            "character": item.get("character", ""),
            "decision": item.get("decision", ""),
            "pending_consequence": item.get("pending_consequence", ""),
            "status": item.get("status", ""),
        }

    return {
        "emotional_traces": [_emo(item) for item in emotional],
        "agency_traces": [_agy(item) for item in agency],
    }


def character_trace_gate_block(traces: dict[str, Any]) -> str:
    """把活跃人物因果痕迹渲染成「人物层反模式门」硬约束（对标 V4 反模式门）。

    只针对义务记忆带来的两个人物层副作用：
    - 情绪痕迹 → 禁止 NO_EMOTIONAL_CONSEQUENCE（情绪无后果消失）；
    - 主体性痕迹 → 禁止 AGENCY_COLLAPSE（选择代价/后果无声消失）。

    门控是条件性的：仅当本集自然触及该人物/该痕迹相关剧情时才强制兑现，
    不触及则不强制，避免把每一集都变成「情绪处理集」、生硬插入。
    这些门只用于幕后校验，禁止被角色复述为规则或口号。
    """
    emotional = traces.get("emotional_traces", [])
    agency = traces.get("agency_traces", [])
    if not emotional and not agency:
        return "（无）"
    lines: list[str] = []
    if emotional:
        for item in emotional:
            character = item.get("character", "")
            residue = item.get("residue", "")
            stake = item.get("unresolved_stake", "")
            lines.append(
                f"- 情绪门：{character} 的「{residue}」仍未消失（仍受威胁：{stake}）。"
                f"若本集触及该人物，这条残留压力必须实际改变 TA 本集至少一个选择、判断或行动"
                f"（如因不信任而拒绝合作、因怕失去而隐瞒、因愧疚而回避正面回答、因怨气而刻意拖延），"
                f"不能只停留在表情、台词或内心波动上；不得无触发归零、不得用一句「没事/过去了」化解。"
                f"本集不触及则可自然淡出。"
            )
    if agency:
        for item in agency:
            character = item.get("character", "")
            decision = item.get("decision", "")
            cost = item.get("accepted_cost", "")
            consequence = item.get("pending_consequence", "")
            lines.append(
                f"- 主体性门：{character} 的选择「{decision}」已接受代价「{cost}」、后果「{consequence}」尚未落地。"
                f"若本集触及该选择，其代价/后果必须实际落地或被明确重新议价，"
                f"不得无声消失、不得被流程或他人自动代偿；本集不触及则可暂缓。"
            )
    return "\n".join(lines)


def build_character_trace_extraction_prompt(
    *,
    story: dict[str, Any],
    episode_id: str,
    script: str,
    emotional: list[dict[str, Any]],
    agency: list[dict[str, Any]],
    mode: str = "both",
) -> str:
    characters = [{"name": item["name"], "role": item["role"]} for item in story["characters"]]
    active = extraction_view(emotional, agency)
    if mode not in {"both", "emotion", "agency"}:
        raise ValueError(f"未知人物痕迹抽取模式：{mode}")
    want_emotion = mode in {"both", "emotion"}
    want_agency = mode in {"both", "agency"}
    emotion_section = '''  "new_emotional_traces": [{
    "character": "人物名",
    "trigger_event": "本集发生的具体事件",
    "appraisal": "该人物认为此事意味着什么",
    "residue": "集末仍未消失的情绪压力及指向对象",
    "unresolved_stake": "仍可能失去或被威胁的东西",
    "last_observed_effect": "已在剧本中出现的行为影响；没有则空字符串",
    "evidence": "剧本逐字证据"
  }],
  "emotional_updates": [{
    "trace_id": "上一集记忆中的 trace_id，必须逐字复制（如 EMO_ 开头的那串 ID）",
    "event": "TRANSFORMED|SETTLED",
    "evidence": "本集具体动作或台词"
  }],'''
    agency_section = '''  "new_agency_traces": [{
    "character": "人物名",
    "decision": "人物亲自作出的选择",
    "motivation_source": "选择来自人物的目标/恐惧/责任/关系",
    "rejected_alternative": "人物明确放弃或拒绝的另一条可行路",
    "accepted_cost": "人物选择时已接受的风险/代价",
    "pending_consequence": "尚未落地的后果；没有则空字符串",
    "last_observed_effect": "该选择已造成的可观察影响",
    "evidence": "剧本逐字证据"
  }],
  "agency_updates": [{
    "trace_id": "上一集记忆中的 trace_id，必须逐字复制（如 AGY_ 开头的那串 ID）",
    "event": "REVISED|PAID|SETTLED",
    "evidence": "本集具体动作或台词"
  }]'''
    sections: list[str] = []
    if want_emotion:
        sections.append(emotion_section)
    if want_agency:
        sections.append(agency_section)
    output_format = "{\n" + "\n".join(sections) + "\n}"
    mode_rule = {
        "both": "1. new_emotional_traces / new_agency_traces 每集默认 0 条，各最多 1 条。",
        "emotion": "1. 本模式只记录情绪痕迹：new_emotional_traces 默认 0 条、最多 1 条；new_agency_traces 与 agency_updates 必须输出空数组 []。",
        "agency": "1. 本模式只记录主体性痕迹：new_agency_traces 默认 0 条、最多 1 条；new_emotional_traces 与 emotional_updates 必须输出空数组 []。",
    }[mode]
    return f"""你是连续剧人物因果痕迹标注器。只根据当前集剧本记录「已经发生、且会影响后续的人物因果痕迹」，不修复、不补写、不评价文笔。只输出合法 JSON，不要 Markdown。

故事：{story['title']}
人物：{json.dumps(characters, ensure_ascii=False)}
当前集：{episode_id}

上一集已记录的活跃人物因果痕迹：
{json.dumps(active, ensure_ascii=False)}

当前集剧本：
{script}

输出格式：
{output_format}

严格规则：
{mode_rule}
2. emotional_trace 仅在同时满足时新增：有明确触发事件；人物有可识别 appraisal；情绪/压力在集末仍未消失；它可能改变后续判断、信任、风险偏好或选择。普通惊讶、短暂生气、泛化氛围不记录。
3. agency_trace 仅在同时满足时新增：存在至少两个真实可行方向；人物本人作出选择而非接受指令；选择可追溯到人物动机；存在被放弃选项或真实风险/代价。接任务、查资料、交文件、按计划执行不记录。
4. 时间流逝、未被提及、完成外部任务、说一句"没事"、一次道歉都不能 SETTLED。SETTLED 必须有明确的处理或转化证据。
5. 同一人物、同一对象、同一 stake 的痕迹若已存在，优先用 updates 更新或转换，不重复新增。
6. 证据必须来自当前集剧本；拿不准时宁可不新增、不更新。字符串内引用原话用中文引号「」，禁止英文单引号。
7. updates 里的 trace_id 必须从「上一集已记录的活跃人物因果痕迹」中逐字复制那个 trace_id（EMO_/AGY_ 开头的完整字符串），不得自己编造「人物名-描述」之类的 ID；没有可对应的活跃痕迹时，updates 输出空数组 []。
8. 最外层只输出上述对象，不要包在 result/output/data 字段。"""
