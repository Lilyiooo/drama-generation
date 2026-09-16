from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .api_client import FakeLLMClient, OpenAICompatibleClient
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now
from .run_generation import call_and_record, extract_json_value
from .schema import ROOT, load_protocol

AUDIT_VERSION = "1.3"
OBLIGATION_TYPES = {
    "FORESHADOW", "MYSTERY", "PROMISE", "PLAN", "DEADLINE", "RELATIONSHIP_DEBT",
}
RELEVANCE_VALUES = {"NONE", "OPPORTUNITY", "DUE"}
EVENT_VALUES = {
    "NONE", "MENTIONED", "PROGRESSED", "CANDIDATE_FULFILLED",
    "CANDIDATE_FAILED", "CANDIDATE_ABANDONED", "CANDIDATE_CONTRADICTED",
}
DRIFT_VALUES = {"NONE", "MINOR", "MAJOR"}
TERMINAL_VALUES = {
    "FULFILLED", "FAILED_WITH_CONSEQUENCE", "ABANDONED_WITH_CONSEQUENCE", "CONTRADICTED",
}
FINAL_OPEN_VALUES = {"OPEN_VALID", "CENSORED", "DROPPED", "PARTIAL"}


class ObligationAuditFakeClient(FakeLLMClient):
    def complete(
        self,
        messages: list[dict[str, str]],
        settings: dict[str, Any],
        *,
        seed: int | None,
        purpose: str,
    ) -> str:
        if purpose.startswith("obligation_audit_episode"):
            return json.dumps({"new_obligations": [], "obligation_events": []}, ensure_ascii=False)
        if purpose.startswith("obligation_audit_verify"):
            return json.dumps({"decisions": []}, ensure_ascii=False)
        if purpose.startswith("obligation_audit_finalize"):
            return json.dumps({"open_obligation_statuses": []}, ensure_ascii=False)
        return super().complete(messages, settings, seed=seed, purpose=purpose)


def _episode_number(episode_id: str) -> int:
    return int(episode_id[1:])


def _clamp_confidence(value: Any) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return 0.0
    return round(max(0.0, min(1.0, float(value))), 4)


def _bigrams(text: str) -> set[str]:
    clean = "".join(str(text).split())
    return {clean[i:i + 2] for i in range(max(0, len(clean) - 1))}


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _compact_obligation(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "obligation_id": item["obligation_id"],
        "type": item["type"],
        "description": item["description"],
        "participants": item.get("participants", []),
        "required_payoff": item["required_payoff"],
        "introduced_episode": item["introduced_episode"],
        "earliest_payoff_episode": item.get("earliest_payoff_episode"),
        "latest_payoff_episode": item.get("latest_payoff_episode"),
        "importance": item["importance"],
        "terminal_closure_required": item["terminal_closure_required"],
    }


def build_episode_audit_prompt(
    *,
    story: dict[str, Any],
    episode_id: str,
    script: str,
    open_obligations: list[dict[str, Any]],
) -> str:
    story_context = {
        "title": story["title"],
        "genre": story["genre"],
        "series_goal": story["series_goal"],
        "characters": [{"name": c["name"], "role": c["role"]} for c in story["characters"]],
    }
    return f"""你是独立的长篇叙事义务审计器。你不知道剧本来自哪个实验条件，也不得评价文笔。只根据本集剧本与此前已建立的义务，建立可核查的跨集义务账本。只输出合法 JSON 对象，不要 Markdown。

故事上下文：
{json.dumps(story_context, ensure_ascii=False)}

当前集：{episode_id}
当前开放义务：
{json.dumps([_compact_obligation(x) for x in open_obligations], ensure_ascii=False)}

当前集剧本：
{script}

输出格式：
{{
  "new_obligations": [{{
    "type": "FORESHADOW|MYSTERY|PROMISE|PLAN|DEADLINE|RELATIONSHIP_DEBT",
    "description": "本集明确建立、未来必须回应的义务",
    "introduction_evidence": "本集逐字或近逐字证据",
    "participants": [],
    "required_payoff": "未来发生什么才算结清",
    "importance": 1,
    "earliest_payoff_episode": null,
    "latest_payoff_episode": null,
    "terminal_closure_required": false,
    "confidence": 0.0
  }}],
  "obligation_events": [{{
    "obligation_id": "开放义务中的原 ID",
    "relevance": "NONE|OPPORTUNITY|DUE",
    "event": "NONE|MENTIONED|PROGRESSED|CANDIDATE_FULFILLED|CANDIDATE_FAILED|CANDIDATE_ABANDONED|CANDIDATE_CONTRADICTED",
    "evidence": "有事件时引用本集证据；无事件为空串",
    "drift": "NONE|MINOR|MAJOR",
    "confidence": 0.0
  }}]
}}

规则：
1. 对每条当前开放义务必须恰好输出一条 obligation_events；没有触及时用 relevance=NONE,event=NONE。不得遗漏或新增 obligation_id。
2. relevance=OPPORTUNITY 仅指本集人物、对象或行动真正触及该义务，存在自然推进机会；一般主题相似不算。relevance=DUE 仅指明确期限已到，或当前集已是该义务不可再延迟的决定时刻。
3. event=MENTIONED 只是重提；PROGRESSED 必须有新信息、新行动或状态改变；四种 CANDIDATE_* 只是候选终局，之后还会独立复核。
4. 对每条开放义务都必须主动检查是否已满足 required_payoff。若本集直接回答了 MYSTERY 的问题、揭示了 FORESHADOW 的意义、完成/明确失败/正式放弃了 PLAN、执行或明确违背了 PROMISE、期限后果已落地、关系债已有实质回应，必须输出对应 CANDIDATE_*，不要一律保守写 PROGRESSED。剧情高潮、关系转折、人物和解本身不自动等于兑现；但剧本已经展示人物持有/查阅某文件，可视为「已获得访问或文件」，不要求额外签字、归档或口头宣布完成。例如「查明谁使用鬼卡」在人物身份被直接确认时即候选兑现；「取得档案」在人物已经阅读档案时即候选兑现；不必等待整条主线结局。
5. 新义务默认输出 0 条，通常 0-1 条，最多 2 条。只登记会跨到未来、遗漏后会形成明显叙事悬空的【实质性债务】；普通愿望、下一步调查、约见、获取/审查一份资料、单步动作、集目标、作者要求都不算。
6. 严禁把每集的调查下一步改写为 DEADLINE。DEADLINE 必须同时有剧本明确时间窗口和到期后的不可忽略后果；「明天去问」「稍后取文件」「下集审查」只是日程或待办，不是期限义务。
7. MYSTERY 不得把同一主问题的线索、子问题或调查范围变化重复登记。如果新的问题可被某条开放 MYSTERY/FORESHADOW 覆盖，应标该旧义务 PROGRESSED，而不是创建新义务。
8. PLAN 必须是人物已承诺执行的多步方案，未来需要看到完成、失败、替代或正式放弃；单次拜访、申请、查阅、问询不算。RELATIONSHIP_DEBT 必须来自明确伤害、亏欠、隐瞒或交换，普通摩擦/合作不算。
9. required_payoff 只写义务成立所必需的最小结果，不得擅自增加「并查明幕后者」「并形成实质关联」「并完成归档」等剧本未承诺的附加条件。
10. importance：3=主线/主要关系/不可逆选择；2=跨多集支线；1=短期局部义务。
11. earliest/latest 只有剧本给出明确或可确定的集数时才填 E01-E40，否则 null；不得自行猜测。
12. terminal_closure_required=true 表示该义务在本剧 40 集结束前按故事承诺理应结清；开放结局、下一季空间或刚建立的义务填 false。
13. drift=MINOR 指条件有轻微无依据变化但核心不变；MAJOR 指人物、对象、期限或 required payoff 被无依据改写。
14. introduction_evidence/evidence 必须来自剧本，禁止写判断说明；若证据只表示普通下一步行动，则不要创建义务。"""


def validate_episode_audit(
    result: dict[str, Any],
    *,
    open_ids: set[str],
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("审计结果必须是对象")
    new_items = result.get("new_obligations")
    events = result.get("obligation_events")
    if not isinstance(new_items, list) or len(new_items) > 2:
        raise ValueError("new_obligations 必须是最多 2 条的数组")
    if not isinstance(events, list):
        raise ValueError("obligation_events 必须是数组")

    clean_new: list[dict[str, Any]] = []
    for index, item in enumerate(new_items):
        if not isinstance(item, dict):
            raise ValueError(f"new_obligations[{index}] 必须是对象")
        obligation_type = str(item.get("type", "")).upper()
        if obligation_type not in OBLIGATION_TYPES:
            raise ValueError(f"new_obligations[{index}].type 非法")
        description = str(item.get("description", "")).strip()
        evidence = str(item.get("introduction_evidence", "")).strip()
        payoff = str(item.get("required_payoff", "")).strip()
        if not description or not evidence or not payoff:
            raise ValueError(f"new_obligations[{index}] 缺少描述、证据或兑现标准")
        participants = item.get("participants", [])
        if not isinstance(participants, list):
            participants = []
        importance = item.get("importance", 1)
        if not isinstance(importance, int) or isinstance(importance, bool) or importance not in {1, 2, 3}:
            importance = 1
        earliest = item.get("earliest_payoff_episode")
        latest = item.get("latest_payoff_episode")
        for field_name, value in (("earliest", earliest), ("latest", latest)):
            if value is not None and (not isinstance(value, str) or not value.startswith("E") or not value[1:].isdigit()):
                raise ValueError(f"new_obligations[{index}].{field_name} 非法")
        clean_new.append({
            "type": obligation_type,
            "description": description,
            "introduction_evidence": evidence,
            "participants": [str(x).strip() for x in participants if str(x).strip()],
            "required_payoff": payoff,
            "importance": importance,
            "earliest_payoff_episode": earliest,
            "latest_payoff_episode": latest,
            "terminal_closure_required": bool(item.get("terminal_closure_required", False)),
            "confidence": _clamp_confidence(item.get("confidence")),
        })

    clean_events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"obligation_events[{index}] 必须是对象")
        obligation_id = str(item.get("obligation_id", ""))
        if obligation_id not in open_ids or obligation_id in seen:
            raise ValueError(f"obligation_events[{index}].obligation_id 非法或重复")
        seen.add(obligation_id)
        relevance = str(item.get("relevance", "NONE")).upper()
        event = str(item.get("event", "NONE")).upper()
        drift = str(item.get("drift", "NONE")).upper()
        if relevance not in RELEVANCE_VALUES or event not in EVENT_VALUES or drift not in DRIFT_VALUES:
            raise ValueError(f"obligation_events[{index}] 枚举值非法")
        evidence = str(item.get("evidence", "")).strip()
        if event != "NONE" and not evidence:
            raise ValueError(f"obligation_events[{index}] 有事件但无证据")
        clean_events.append({
            "obligation_id": obligation_id,
            "relevance": relevance,
            "event": event,
            "evidence": evidence,
            "drift": drift,
            "confidence": _clamp_confidence(item.get("confidence")),
        })
    if seen != open_ids:
        raise ValueError(f"obligation_events 未完整覆盖开放义务：缺少 {sorted(open_ids - seen)}")
    return {"new_obligations": clean_new, "obligation_events": clean_events}


def build_verification_prompt(
    *,
    episode_id: str,
    script: str,
    candidates: list[dict[str, Any]],
) -> str:
    payload = [{
        "obligation_id": item["obligation_id"],
        "type": item["type"],
        "description": item["description"],
        "required_payoff": item["required_payoff"],
        "candidate_event": item["candidate_event"],
        "candidate_evidence": item["candidate_evidence"],
    } for item in candidates]
    return f"""你是叙事义务终局事件复核器。只判断候选终局是否有当前集剧本的直接证据，不评价文笔，不参考实验条件。只输出合法 JSON。

当前集：{episode_id}
候选终局：
{json.dumps(payload, ensure_ascii=False)}

当前集剧本：
{script}

输出：
{{"decisions":[{{"obligation_id":"原ID","verdict":"CONFIRMED|REJECTED","terminal_event":"FULFILLED|FAILED_WITH_CONSEQUENCE|ABANDONED_WITH_CONSEQUENCE|CONTRADICTED|NONE","evidence":"剧本直接证据或拒绝理由","confidence":0.0}}]}}

每个候选必须恰好输出一条。required_payoff 的最小必要结果已经发生即可确认，不要求额外的签字、归档、总结台词或仪式性闭环：人物已在查阅某档案，足以证明已获得访问；人物已拿到并使用记录，足以证明已取得记录。仅提到、只有准备动作、部分推进、关系转折、临近期限仍应拒绝。失败/放弃必须同时有明确失败/撤回和可见后果；CONTRADICTED 必须是关键人物、对象、期限或兑现条件被无依据改写。"""


def validate_verification(result: dict[str, Any], candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    decisions = result.get("decisions") if isinstance(result, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("decisions 必须是数组")
    clean: dict[str, dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("decision 必须是对象")
        obligation_id = str(item.get("obligation_id", ""))
        if obligation_id not in candidate_ids or obligation_id in clean:
            raise ValueError("decision ID 非法或重复")
        verdict = str(item.get("verdict", "")).upper()
        terminal_event = str(item.get("terminal_event", "NONE")).upper()
        if verdict not in {"CONFIRMED", "REJECTED"}:
            raise ValueError("verdict 非法")
        if terminal_event not in TERMINAL_VALUES | {"NONE"}:
            raise ValueError("terminal_event 非法")
        if verdict == "CONFIRMED" and terminal_event == "NONE":
            raise ValueError("确认事件不能为 NONE")
        if verdict == "REJECTED":
            terminal_event = "NONE"
        clean[obligation_id] = {
            "verdict": verdict,
            "terminal_event": terminal_event,
            "evidence": str(item.get("evidence", "")).strip(),
            "confidence": _clamp_confidence(item.get("confidence")),
        }
    if set(clean) != candidate_ids:
        raise ValueError("decisions 未完整覆盖候选")
    return clean


def build_finalize_prompt(
    *,
    story: dict[str, Any],
    open_obligations: list[dict[str, Any]],
) -> str:
    payload = []
    for item in open_obligations:
        relevant_history = [
            {
                "episode_id": event.get("episode_id"),
                "relevance": event.get("relevance"),
                "event": event.get("event"),
                "evidence": event.get("evidence", ""),
                "drift": event.get("drift", "NONE"),
            }
            for event in item.get("history", [])
            if event.get("relevance") != "NONE" or event.get("event") != "NONE" or event.get("drift") != "NONE"
        ]
        payload.append({
            **_compact_obligation(item),
            "introduction_evidence": item.get("introduction_evidence", ""),
            "relevant_history": relevant_history[-8:],
        })
    return f"""你是长篇叙事义务终局审计器。全剧已到 E40。根据系列目标与每条义务的审计历史，判断尚未结清的义务在结尾属于合理开放、右删失、被遗忘或部分完成。只输出合法 JSON。

系列目标：{story['series_goal']}
尚未结清义务：
{json.dumps(payload, ensure_ascii=False)}

输出：
{{"open_obligation_statuses":[{{"obligation_id":"原ID","status":"OPEN_VALID|CENSORED|DROPPED|PARTIAL","reason":"基于历史的简短理由","confidence":0.0}}]}}

判定：
- OPEN_VALID：系列目标明确允许保留，或开放结局仍保持一致；不扣兑现分。
- CENSORED：义务建立过晚、没有足够观察窗口，或尚未出现自然兑现机会；不进入兑现率分母。
- DROPPED：明确期限/不可再延迟时刻已过，或出现直接处理机会却无声遗忘。
- PARTIAL：已有实质推进，但到应处理的时点仍未完成。
不得因为 E40 到来就把所有开放义务判为 DROPPED。每个 ID 必须恰好输出一次。
注意：JSON 语法要求所有字符串值必须用英文双引号 " 包裹。值内部如需引用原话或词语，一律用中文引号「」；禁止用英文单引号 ' 包裹字符串，也禁止在值内部出现英文双引号 "，否则 JSON 无法解析。"""


def validate_finalize(result: dict[str, Any], open_ids: set[str]) -> dict[str, dict[str, Any]]:
    statuses = result.get("open_obligation_statuses") if isinstance(result, dict) else None
    if isinstance(statuses, dict):
        # 容错：模型偶发输出 {obligation_id: status} 或 {obligation_id: {status, reason}} 映射
        converted: list[dict[str, Any]] = []
        for key, value in statuses.items():
            if isinstance(value, str):
                converted.append({"obligation_id": key, "status": value})
            elif isinstance(value, dict):
                converted.append({"obligation_id": key, **value})
            else:
                converted.append({"obligation_id": key})
        statuses = converted
    if not isinstance(statuses, list):
        raise ValueError("open_obligation_statuses 必须是数组或映射")
    clean: dict[str, dict[str, Any]] = {}
    for item in statuses:
        if not isinstance(item, dict):
            raise ValueError("终局状态必须是对象")
        obligation_id = str(item.get("obligation_id", ""))
        status = str(item.get("status", "")).upper()
        if obligation_id not in open_ids or obligation_id in clean or status not in FINAL_OPEN_VALUES:
            raise ValueError("终局状态 ID 或枚举非法")
        clean[obligation_id] = {
            "status": status,
            "reason": str(item.get("reason", "")).strip(),
            "confidence": _clamp_confidence(item.get("confidence")),
        }
    if set(clean) != open_ids:
        raise ValueError("终局状态未完整覆盖开放义务")
    return clean


def _repair_unescaped_quotes(text: str) -> str:
    """修复 JSON 字符串值内未转义的英文双引号（LLM 偶发把 `"保人"` 写成英文引号导致顶层解析失败）。"""
    result: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                result.append(char)
                escaped = False
                continue
            if char == "\\":
                result.append(char)
                escaped = True
                continue
            if char == '"':
                # 判断是字符串结束还是值内未转义引号：后面紧跟 JSON 结构符则为结束，否则视为值内引号转义
                rest = text[index + 1:index + 1 + 8].lstrip()
                if rest and rest[0] in ",}]:\n":
                    result.append(char)
                    in_string = False
                else:
                    result.append('\\"')
                continue
            result.append(char)
        else:
            if char == '"':
                in_string = True
            result.append(char)
    return "".join(result)


def _call_structured(
    *,
    client: Any,
    prompt: str,
    settings: dict[str, Any],
    seed: int,
    purpose_prefix: str,
    job: dict[str, Any],
    output_dir: Path,
    validator: Any,
    attempts: int = 4,
) -> Any:
    last_error = ""
    for attempt in range(1, attempts + 1):
        retry = "" if attempt == 1 else f"\n\n上次输出未通过校验：{last_error}。请重新输出完整合法 JSON。"
        raw = call_and_record(
            client,
            messages=[{"role": "user", "content": prompt + retry}],
            settings=settings,
            seed=seed + attempt - 1,
            purpose=f"{purpose_prefix}_attempt_{attempt}",
            job=job,
            output_dir=output_dir,
        )
        candidates = [raw]
        try:
            candidates.append(_repair_unescaped_quotes(raw))
        except Exception:
            pass
        try:
            candidates.append(raw.replace("'", '"'))
        except Exception:
            pass
        for candidate in candidates:
            try:
                return validator(extract_json_value(candidate))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = str(exc)
    raise ValueError(f"{job['job_id']} 连续 {attempts} 次审计失败：{last_error}")


def _new_obligation_id(sequence: int) -> str:
    """轨迹内匿名 ID；绝不向审计模型泄露 condition/story/run。"""
    return f"O{sequence:04d}"


def _job_run_id(row: dict[str, Any]) -> str:
    if row.get("run_id"):
        return str(row["run_id"])
    return str(row["trajectory_id"]).split("__")[-1]


def audit_run(
    *,
    run_dir: Path,
    output_dir: Path,
    fake_api: bool,
    execute_api: bool,
    stories_filter: set[str] | None,
    runs_filter: set[str] | None,
    limit_trajectories: int | None,
) -> int:
    config, stories, plans, _, _ = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止")
    client = ObligationAuditFakeClient() if fake_api else OpenAICompatibleClient.from_environment("evaluation")
    settings = dict(config["evaluation"])
    settings["temperature"] = 0.0
    settings["max_output_tokens"] = max(6144, int(settings.get("max_output_tokens", 6144)))
    stories_by_id = {item["story_id"]: item for item in stories}
    plans_by_key = {(item["story_id"], item["episode_id"]): item for item in plans}
    generations = read_jsonl(run_dir / "generations.jsonl")
    if stories_filter:
        generations = [row for row in generations if row["story_id"] in stories_filter]
    if runs_filter:
        generations = [row for row in generations if _job_run_id(row) in runs_filter]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generations:
        grouped[row["trajectory_id"]].append(row)
    trajectory_ids = sorted(grouped)
    if limit_trajectories is not None:
        trajectory_ids = trajectory_ids[:limit_trajectories]

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = output_dir / "obligation_audit_episodes.jsonl"
    summary_path = output_dir / "obligation_audit_trajectories.jsonl"
    existing_episodes = index_jsonl(episode_path, "job_id")
    existing_summaries = index_jsonl(summary_path, "trajectory_id")
    completed = 0
    expected_episode_ids = list(config["episode_ids"])
    evaluator_model = getattr(client, "model", "fake")
    settings_hash = stable_hash(settings)

    for trajectory_id in trajectory_ids:
        rows = sorted(grouped[trajectory_id], key=lambda row: _episode_number(row["episode_id"]))
        if [row["episode_id"] for row in rows] != expected_episode_ids:
            raise ValueError(f"{trajectory_id} 集数不完整或顺序异常，禁止按 E40 终局审计")
        first = rows[0]
        story = stories_by_id[first["story_id"]]
        trajectory_hash = stable_hash([row.get("script_hash", stable_hash(row["script"])) for row in rows])
        existing_summary = existing_summaries.get(trajectory_id)
        if existing_summary:
            if (
                existing_summary.get("audit_version") != AUDIT_VERSION
                or existing_summary.get("trajectory_hash") != trajectory_hash
                or existing_summary.get("evaluator_model") != evaluator_model
                or existing_summary.get("settings_hash") != settings_hash
            ):
                raise ValueError(f"{trajectory_id} 已有审计摘要与当前输入不一致，请使用新 output-dir")
            completed += 1
            continue
        obligations: dict[str, dict[str, Any]] = {}
        open_ids: list[str] = []
        next_obligation_sequence = 1
        resume_prefix = True

        for generation in rows:
            existing = existing_episodes.get(generation["job_id"])
            current_script_hash = generation.get("script_hash", stable_hash(generation["script"]))
            if existing and resume_prefix:
                if (
                    existing.get("audit_version") != AUDIT_VERSION
                    or existing.get("script_hash") != current_script_hash
                    or existing.get("evaluator_model") != evaluator_model
                    or existing.get("settings_hash") != settings_hash
                ):
                    raise ValueError(f"{generation['job_id']} 已有审计快照与当前输入不一致，请使用新 output-dir")
                obligations = {item["obligation_id"]: item for item in existing["obligations_after"]}
                open_ids = [item["obligation_id"] for item in existing["obligations_after"] if item["status"] == "OPEN"]
                existing_numbers = [int(item["obligation_id"][1:]) for item in obligations.values() if item["obligation_id"].startswith("O")]
                next_obligation_sequence = max(existing_numbers, default=0) + 1
                continue
            if existing and not resume_prefix:
                raise ValueError(f"{generation['job_id']} 位于断点后的旧快照，禁止非连续恢复；请使用新 output-dir")
            resume_prefix = False

            open_before = [obligations[item_id] for item_id in open_ids]
            prompt = build_episode_audit_prompt(
                story=story,
                episode_id=generation["episode_id"],
                script=generation["script"],
                open_obligations=open_before,
            )
            audit = _call_structured(
                client=client,
                prompt=prompt,
                settings=settings,
                seed=int(generation.get("seed", 0)) + 5100,
                purpose_prefix="obligation_audit_episode_v1",
                job=generation,
                output_dir=output_dir,
                validator=lambda value, ids=set(open_ids): validate_episode_audit(value, open_ids=ids),
            )

            events_by_id = {item["obligation_id"]: item for item in audit["obligation_events"]}
            candidates: list[dict[str, Any]] = []
            for obligation_id in list(open_ids):
                event = events_by_id[obligation_id]
                history_item = {"episode_id": generation["episode_id"], **copy.deepcopy(event)}
                obligations[obligation_id].setdefault("history", []).append(history_item)
                if event["event"].startswith("CANDIDATE_"):
                    candidates.append({
                        **_compact_obligation(obligations[obligation_id]),
                        "candidate_event": event["event"],
                        "candidate_evidence": event["evidence"],
                    })

            if candidates:
                verification_prompt = build_verification_prompt(
                    episode_id=generation["episode_id"],
                    script=generation["script"],
                    candidates=candidates,
                )
                candidate_ids = {item["obligation_id"] for item in candidates}
                decisions = _call_structured(
                    client=client,
                    prompt=verification_prompt,
                    settings=settings,
                    seed=int(generation.get("seed", 0)) + 6100,
                    purpose_prefix="obligation_audit_verify_v1",
                    job=generation,
                    output_dir=output_dir,
                    validator=lambda value, ids=candidate_ids: validate_verification(value, ids),
                )
                for obligation_id, decision in decisions.items():
                    history_item = obligations[obligation_id]["history"][-1]
                    history_item["verification"] = decision
                    if decision["verdict"] == "CONFIRMED":
                        obligations[obligation_id]["status"] = decision["terminal_event"]
                        obligations[obligation_id]["resolution_episode"] = generation["episode_id"]
                        open_ids.remove(obligation_id)
                    else:
                        history_item["event"] = "PROGRESSED" if history_item["relevance"] != "NONE" else "NONE"

            for candidate in audit["new_obligations"]:
                description = candidate["description"]
                if any(_overlap(description, obligations[item_id]["description"]) > 0.6 for item_id in open_ids):
                    continue
                obligation_id = _new_obligation_id(next_obligation_sequence)
                next_obligation_sequence += 1
                obligations[obligation_id] = {
                    "obligation_id": obligation_id,
                    **candidate,
                    "introduced_episode": generation["episode_id"],
                    "status": "OPEN",
                    "resolution_episode": None,
                    "history": [],
                }
                open_ids.append(obligation_id)

            record = {
                "audit_version": AUDIT_VERSION,
                "source_run_dir": str(run_dir),
                "evaluator_model": evaluator_model,
                "settings_hash": settings_hash,
                "job_id": generation["job_id"],
                "trajectory_id": trajectory_id,
                "story_id": generation["story_id"],
                "run_id": _job_run_id(generation),
                "episode_id": generation["episode_id"],
                "script_hash": generation.get("script_hash", stable_hash(generation["script"])),
                "plan_hash": stable_hash(plans_by_key[(generation["story_id"], generation["episode_id"])]),
                "new_obligations": [obligations[item_id] for item_id in open_ids if obligations[item_id]["introduced_episode"] == generation["episode_id"]],
                "events": audit["obligation_events"],
                "obligations_after": [copy.deepcopy(obligations[item_id]) for item_id in obligations],
                "open_count": len(open_ids),
                "created_at": utc_now(),
            }
            append_jsonl(episode_path, record)
            existing_episodes[generation["job_id"]] = record

        open_obligations = [obligations[item_id] for item_id in open_ids]
        if open_obligations:
            final_job = dict(rows[-1])
            final_job["job_id"] = f"{trajectory_id}__FINAL"
            final_prompt = build_finalize_prompt(story=story, open_obligations=open_obligations)
            open_id_set = set(open_ids)
            final_statuses = _call_structured(
                client=client,
                prompt=final_prompt,
                settings=settings,
                seed=int(rows[-1].get("seed", 0)) + 7100,
                purpose_prefix="obligation_audit_finalize_v1",
                job=final_job,
                output_dir=output_dir,
                validator=lambda value, ids=open_id_set: validate_finalize(value, ids),
            )
            for obligation_id, final_status in final_statuses.items():
                obligations[obligation_id]["status"] = final_status["status"]
                obligations[obligation_id]["final_reason"] = final_status["reason"]
                obligations[obligation_id]["final_confidence"] = final_status["confidence"]

        summary = {
            "audit_version": AUDIT_VERSION,
            "source_run_dir": str(run_dir),
            "evaluator_model": evaluator_model,
            "settings_hash": settings_hash,
            "trajectory_id": trajectory_id,
            "story_id": first["story_id"],
            "run_id": _job_run_id(first),
            "condition": first.get("condition"),
            "story_hash": stable_hash(story),
            "trajectory_hash": trajectory_hash,
            "episode_count": len(rows),
            "obligations": list(obligations.values()),
            "created_at": utc_now(),
        }
        append_jsonl(summary_path, summary)
        existing_summaries[trajectory_id] = summary
        completed += 1
        print(f"[audit] {trajectory_id}: obligations={len(obligations)}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fake-api", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--stories", type=str)
    parser.add_argument("--runs", type=str)
    parser.add_argument("--limit-trajectories", type=int)
    args = parser.parse_args()
    stories_filter = {x.strip() for x in args.stories.split(",") if x.strip()} if args.stories else None
    runs_filter = {x.strip() for x in args.runs.split(",") if x.strip()} if args.runs else None
    count = audit_run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        fake_api=args.fake_api,
        execute_api=args.execute_api,
        stories_filter=stories_filter,
        runs_filter=runs_filter,
        limit_trajectories=args.limit_trajectories,
    )
    print(f"audited_trajectories={count} fake_api={args.fake_api}")


if __name__ == "__main__":
    main()
