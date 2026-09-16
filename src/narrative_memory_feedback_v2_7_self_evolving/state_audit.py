"""Independent trajectory state audit.

The auditor reconstructs a state ledger from scripts and the story's initial state.
It deliberately does not read ``state_updates.jsonl`` while deciding what happened;
that internal ledger is only used later by ``analyze_state_continuity`` for
precision/recall diagnostics.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .api_client import FakeLLMClient, OpenAICompatibleClient
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now
from .obligation_audit import _call_structured, _clamp_confidence, _job_run_id
from .schema import ROOT, load_protocol


AUDIT_VERSION = "1.0"

STATE_DOMAINS = {
    "WORLD_FACT",
    "CHARACTER",
    "KNOWLEDGE",
    "RESOURCE_EVIDENCE",
    "GOAL",
    "UNKNOWN",
    "TIMELINE",
    "RELATIONSHIP",
    "CAUSAL_THREAD",
}
PROSPECTIVE_DOMAINS = {"GOAL", "UNKNOWN", "CAUSAL_THREAD"}
CERTAINTY_VALUES = {"CONFIRMED", "REPORTED", "INFERRED", "UNKNOWN"}
STATE_EVENTS = {"UPDATED", "RESOLVED", "RETRACTED", "TRANSFERRED", "CONSUMED", "LOST"}
CLOSING_EVENTS = {"RESOLVED", "RETRACTED", "CONSUMED", "LOST"}
ISSUE_TYPES = {
    "FACT_CONTRADICTION",
    "UNSUPPORTED_TRANSITION",
    "KNOWLEDGE_LEAK",
    "RESOURCE_DISCONTINUITY",
    "TIMELINE_CONFLICT",
    "IDENTITY_ROLE_CONFLICT",
    "RELATIONSHIP_RESET",
    "STALE_STATE",
    "MISSING_CAUSAL_LINK",
}
SEVERITY_VALUES = {"MINOR", "MAJOR", "CRITICAL"}
FINAL_OPEN_VALUES = {"OPEN_VALID", "CENSORED", "STALE", "PARTIAL"}

FIELD_DOMAIN = {
    "character_state": "CHARACTER",
    "relationship_state": "RELATIONSHIP",
    "known_information": "KNOWLEDGE",
    "unknown_information": "UNKNOWN",
    "confirmed_facts": "WORLD_FACT",
    "unresolved_threads": "CAUSAL_THREAD",
    "resources_and_evidence": "RESOURCE_EVIDENCE",
    "current_goals": "GOAL",
    "timeline": "TIMELINE",
}


def _episode_number(episode_id: str) -> int:
    return int(episode_id[1:])


def _bigrams(text: str) -> set[str]:
    clean = "".join(str(text).split()).lower()
    return {clean[index:index + 2] for index in range(max(0, len(clean) - 1))}


def _overlap(left: str, right: str) -> float:
    a, b = _bigrams(left), _bigrams(right)
    return len(a & b) / len(a | b) if a and b else 0.0


def _state_description(subject: str, attribute: str, value: str) -> str:
    if subject and subject != "STORY":
        return f"{subject}：{attribute}={value}"
    return f"{attribute}={value}"


def _new_state_id(sequence: int) -> str:
    return f"ST{sequence:04d}"


def _new_issue_id(sequence: int) -> str:
    return f"SI{sequence:04d}"


def _initial_certainty(domain: str) -> str:
    return "UNKNOWN" if domain == "UNKNOWN" else "CONFIRMED"


def seed_initial_ledger(story: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert protocol initial_state into an anonymous external audit ledger."""
    ledger: dict[str, dict[str, Any]] = {}
    sequence = 1

    def add(domain: str, subject: str, attribute: str, value: Any) -> None:
        nonlocal sequence
        text = str(value).strip()
        if not text:
            return
        state_id = _new_state_id(sequence)
        sequence += 1
        ledger[state_id] = {
            "state_id": state_id,
            "domain": domain,
            "subject": subject,
            "attribute": attribute,
            "value": text,
            "description": _state_description(subject, attribute, text),
            "certainty": _initial_certainty(domain),
            "importance": 2,
            "status": "ACTIVE",
            "introduced_episode": "E00",
            "last_update_episode": "E00",
            "closed_episode": None,
            "introduction_evidence": f"故事初始状态：{text}",
            "history": [{"episode_id": "E00", "event": "ESTABLISHED", "value": text}],
        }

    for field, domain in FIELD_DOMAIN.items():
        value = story["initial_state"].get(field, [])
        if isinstance(value, dict):
            for subject, entries in value.items():
                items = entries if isinstance(entries, list) else [entries]
                for index, item in enumerate(items, start=1):
                    add(domain, str(subject), f"{field}_{index:02d}", item)
        elif isinstance(value, list):
            for index, item in enumerate(value, start=1):
                add(domain, "STORY", f"{field}_{index:02d}", item)
    return ledger


def _compact_state(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "state_id": item["state_id"],
        "domain": item["domain"],
        "subject": item.get("subject", ""),
        "attribute": item.get("attribute", ""),
        "value": item.get("value", ""),
        "description": item.get("description", ""),
        "certainty": item.get("certainty", "CONFIRMED"),
        "introduced_episode": item.get("introduced_episode"),
        "last_update_episode": item.get("last_update_episode"),
        "status": item.get("status", "ACTIVE"),
        "closed_episode": item.get("closed_episode"),
    }


def build_episode_audit_prompt(
    *,
    story: dict[str, Any],
    episode_id: str,
    script: str,
    ledger_states: list[dict[str, Any]],
) -> str:
    context = {
        "title": story["title"],
        "genre": story["genre"],
        "series_goal": story["series_goal"],
        "characters": [{"name": item["name"], "role": item["role"]} for item in story["characters"]],
    }
    return f"""你是独立的长篇连续剧状态审计器。你不知道实验条件，也不得评价文笔。你只根据故事初始设定、此前由本审计器从剧本重建的开放状态、以及当前集剧本，维护可核查的外部状态账本。禁止参考生成系统的内部 state_updates。只输出合法 JSON，不要 Markdown。

故事上下文：
{json.dumps(context, ensure_ascii=False)}

当前集：{episode_id}
当前外部状态账本（ACTIVE 可更新；已关闭状态仅供检查无依据复活、矛盾和重复消费）：
{json.dumps([_compact_state(item) for item in ledger_states], ensure_ascii=False)}

当前集剧本：
{script}

输出格式：
{{
  "new_states": [{{
    "domain": "WORLD_FACT|CHARACTER|KNOWLEDGE|RESOURCE_EVIDENCE|GOAL|UNKNOWN|TIMELINE|RELATIONSHIP|CAUSAL_THREAD",
    "subject": "人物、组织、物件或 STORY",
    "attribute": "稳定、可跨集比较的属性名",
    "value": "本集结束后的值",
    "description": "不依赖上下文也能理解的完整状态句",
    "certainty": "CONFIRMED|REPORTED|INFERRED|UNKNOWN",
    "importance": 1,
    "evidence": "本集具体动作、台词或场面证据"
  }}],
  "state_events": [{{
    "state_id": "当前开放状态中的原 ID",
    "event": "UPDATED|RESOLVED|RETRACTED|TRANSFERRED|CONSUMED|LOST",
    "new_value": "UPDATED/TRANSFERRED 后的新值，否则空字符串",
    "evidence": "本集具体证据",
    "cause_state_ids": []
  }}],
  "candidate_issues": [{{
    "issue_type": "FACT_CONTRADICTION|UNSUPPORTED_TRANSITION|KNOWLEDGE_LEAK|RESOURCE_DISCONTINUITY|TIMELINE_CONFLICT|IDENTITY_ROLE_CONFLICT|RELATIONSHIP_RESET|STALE_STATE|MISSING_CAUSAL_LINK",
    "domains": ["受影响的状态域"],
    "state_ids": ["相关旧状态 ID，可为空"],
    "description": "当前剧本与历史状态之间的具体问题",
    "evidence": "当前集中的冲突证据",
    "severity": "MINOR|MAJOR|CRITICAL",
    "confidence": 0.0
  }}]
}}

严格规则：
1. 只记录本集真正新增或改变、后续仍可能需要知道的状态。普通动作、气氛、修辞、作者要求、剧集目标和写作策略不进入账本。new_states 默认 0-6 条，最多 12 条。
2. 同一 ACTIVE 状态槽已存在时不得重复新建；应使用 state_events 更新原 state_id，且 state_events 只能引用 ACTIVE ID。UPDATED 表示同一主体同一属性有剧本支持的新值；TRANSFERRED 表示资源、证据、权限或责任发生持有人转移。已关闭状态无依据重新出现时不要重新激活，应提出相应 candidate_issue，并可在 state_ids 引用该历史 ID。
3. RESOLVED 只用于目标、未知和未决因果线程已经得到直接结果；RETRACTED 用于旧事实被明确推翻；CONSUMED/LOST 用于资源或证据确实被消耗或失去。仅提及、接近完成、情绪转折不算关闭。
4. KNOWLEDGE 必须写清知道信息的人。观众知道、另一人物知道，不等于当前人物知道。人物使用此前没有获得的信息时提出 KNOWLEDGE_LEAK 候选。
5. RESOURCE_EVIDENCE 必须保持来源、持有人、转移和消耗连续。无来源出现、失去后复活或重复使用已消耗资源时提出 RESOURCE_DISCONTINUITY。
6. TIMELINE 检查日期、期限、年龄、行程和先后顺序。只有无法被合理省略解释的冲突才提出候选。
7. 人物职位、身体能力、权限、道德边界和稳定身份的变化必须有触发；否则提出 IDENTITY_ROLE_CONFLICT 或 UNSUPPORTED_TRANSITION。
8. 关系可以缓慢变化，但既有伤害、信任崩塌、权力边界不得在无回应、无代价时归零；无依据恢复时提出 RELATIONSHIP_RESET。
9. MISSING_CAUSAL_LINK 指当前关键结果缺少此前条件、人物行动或资源来源，不是一般意义上的戏不好看。
10. candidate_issues 只是候选，后续会独立复核。没有明确跨集问题时输出空数组；不要把正常、有证据的状态变化误报为矛盾。
11. evidence 必须来自当前集剧本；字符串内引用原话使用中文引号，禁止值内部使用英文双引号。
12. 不要求逐一重述未变化状态。最外层只能输出上述三个数组。"""


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def validate_episode_audit(
    result: dict[str, Any], *, active_ids: set[str], known_ids: set[str] | None = None
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("状态审计结果必须是对象")
    new_states = result.get("new_states")
    events = result.get("state_events")
    issues = result.get("candidate_issues")
    if not isinstance(new_states, list) or len(new_states) > 12:
        raise ValueError("new_states 必须是最多 12 条的数组")
    if not isinstance(events, list) or len(events) > 12:
        raise ValueError("state_events 必须是最多 12 条的数组")
    if not isinstance(issues, list) or len(issues) > 8:
        raise ValueError("candidate_issues 必须是最多 8 条的数组")

    all_known_ids = known_ids if known_ids is not None else active_ids
    cleaned_new: list[dict[str, Any]] = []
    for index, item in enumerate(new_states):
        if not isinstance(item, dict):
            raise ValueError(f"new_states[{index}] 必须是对象")
        domain = str(item.get("domain", "")).upper()
        certainty = str(item.get("certainty", "CONFIRMED")).upper()
        if domain not in STATE_DOMAINS or certainty not in CERTAINTY_VALUES:
            raise ValueError(f"new_states[{index}] 的 domain/certainty 非法")
        subject = str(item.get("subject", "STORY")).strip() or "STORY"
        attribute = str(item.get("attribute", "")).strip()
        value = str(item.get("value", "")).strip()
        description = str(item.get("description", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not attribute or not value or not description or not evidence:
            raise ValueError(f"new_states[{index}] 缺少 attribute/value/description/evidence")
        importance = item.get("importance", 1)
        if not isinstance(importance, int) or isinstance(importance, bool) or importance not in {1, 2, 3}:
            importance = 1
        cleaned_new.append({
            "domain": domain,
            "subject": subject,
            "attribute": attribute,
            "value": value,
            "description": description,
            "certainty": certainty,
            "importance": importance,
            "evidence": evidence,
        })

    cleaned_events: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise ValueError(f"state_events[{index}] 必须是对象")
        state_id = str(item.get("state_id", ""))
        event = str(item.get("event", "")).upper()
        if state_id not in active_ids or state_id in seen_events or event not in STATE_EVENTS:
            raise ValueError(f"state_events[{index}] 的 ID/event 非法或重复")
        evidence = str(item.get("evidence", "")).strip()
        new_value = str(item.get("new_value", "")).strip()
        if not evidence or (event in {"UPDATED", "TRANSFERRED"} and not new_value):
            raise ValueError(f"state_events[{index}] 缺少 evidence 或 new_value")
        cause_ids = _clean_string_list(item.get("cause_state_ids"))
        if any(value not in all_known_ids for value in cause_ids):
            raise ValueError(f"state_events[{index}].cause_state_ids 含未知 ID")
        cleaned_events.append({
            "state_id": state_id,
            "event": event,
            "new_value": new_value,
            "evidence": evidence,
            "cause_state_ids": cause_ids,
        })
        seen_events.add(state_id)

    cleaned_issues: list[dict[str, Any]] = []
    for index, item in enumerate(issues):
        if not isinstance(item, dict):
            raise ValueError(f"candidate_issues[{index}] 必须是对象")
        issue_type = str(item.get("issue_type", "")).upper()
        severity = str(item.get("severity", "MAJOR")).upper()
        domains = [value.upper() for value in _clean_string_list(item.get("domains"))]
        state_ids = _clean_string_list(item.get("state_ids"))
        if issue_type not in ISSUE_TYPES or severity not in SEVERITY_VALUES:
            raise ValueError(f"candidate_issues[{index}] 的类型或严重度非法")
        if not domains or any(value not in STATE_DOMAINS for value in domains):
            raise ValueError(f"candidate_issues[{index}].domains 非法")
        if any(value not in all_known_ids for value in state_ids):
            raise ValueError(f"candidate_issues[{index}].state_ids 含未知 ID")
        description = str(item.get("description", "")).strip()
        evidence = str(item.get("evidence", "")).strip()
        if not description or not evidence:
            raise ValueError(f"candidate_issues[{index}] 缺少描述或证据")
        cleaned_issues.append({
            "candidate_id": f"C{index + 1:02d}",
            "issue_type": issue_type,
            "domains": list(dict.fromkeys(domains)),
            "state_ids": state_ids,
            "description": description,
            "evidence": evidence,
            "severity": severity,
            "confidence": _clamp_confidence(item.get("confidence")),
        })
    return {"new_states": cleaned_new, "state_events": cleaned_events, "candidate_issues": cleaned_issues}


def build_verification_prompt(
    *, episode_id: str, script: str, candidates: list[dict[str, Any]], ledger: dict[str, dict[str, Any]]
) -> str:
    payload = []
    for item in candidates:
        payload.append({
            **item,
            "prior_states": [_compact_state(ledger[state_id]) for state_id in item["state_ids"] if state_id in ledger],
        })
    return f"""你是跨集状态问题复核器。只判断候选问题是否被当前剧本与此前外部状态直接支持，不评价文笔，不参考实验条件。正常且有触发的状态变化必须拒绝；不能因为剧本省略了无关日常过程就判冲突。只输出合法 JSON。

当前集：{episode_id}
候选问题：
{json.dumps(payload, ensure_ascii=False)}

当前集剧本：
{script}

输出格式：
{{"decisions":[{{"candidate_id":"C01","verdict":"CONFIRMED|REJECTED","evidence":"确认冲突的双边证据或拒绝理由","confidence":0.0}}]}}

每个候选必须恰好输出一条。CONFIRMED 必须指出旧状态和当前剧本为何不能同时成立，或人物为何不可能获得相关知识/资源。仅仅没有再次解释来源、正常时间跳跃、人物有证据的成长与关系变化，都应 REJECTED。"""


def validate_verification(result: dict[str, Any], candidate_ids: set[str]) -> dict[str, dict[str, Any]]:
    decisions = result.get("decisions") if isinstance(result, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("decisions 必须是数组")
    clean: dict[str, dict[str, Any]] = {}
    for item in decisions:
        if not isinstance(item, dict):
            raise ValueError("decision 必须是对象")
        candidate_id = str(item.get("candidate_id", ""))
        verdict = str(item.get("verdict", "")).upper()
        if candidate_id not in candidate_ids or candidate_id in clean or verdict not in {"CONFIRMED", "REJECTED"}:
            raise ValueError("decision ID/verdict 非法或重复")
        clean[candidate_id] = {
            "verdict": verdict,
            "evidence": str(item.get("evidence", "")).strip(),
            "confidence": _clamp_confidence(item.get("confidence")),
        }
    if set(clean) != candidate_ids:
        raise ValueError("decisions 未完整覆盖候选")
    return clean


def build_finalize_prompt(*, story: dict[str, Any], open_states: list[dict[str, Any]]) -> str:
    payload = []
    for item in open_states:
        payload.append({
            **_compact_state(item),
            "importance": item.get("importance", 1),
            "recent_history": item.get("history", [])[-6:],
        })
    return f"""你是长篇状态生命周期终局审计器。全剧已到 E40。只判断尚未关闭的目标、未知和因果线程为何仍开放，不评价文笔。系列目标：{story['series_goal']}

待终审开放状态：
{json.dumps(payload, ensure_ascii=False)}

输出格式：
{{"open_state_statuses":[{{"state_id":"原 ID","status":"OPEN_VALID|CENSORED|STALE|PARTIAL","reason":"基于轨迹状态的理由","confidence":0.0}}]}}

OPEN_VALID=开放结局明确允许继续存在；CENSORED=建立过晚或没有足够观察窗口；STALE=剧本已经得到答案/完成/失效却仍留在开放账本，或应处理时点已过且被遗忘；PARTIAL=有实质推进但尚未完成。不得因为 E40 到来就把所有状态判为 STALE。每个 ID 必须恰好输出一次。"""


def validate_finalize(result: dict[str, Any], open_ids: set[str]) -> dict[str, dict[str, Any]]:
    values = result.get("open_state_statuses") if isinstance(result, dict) else None
    if not isinstance(values, list):
        raise ValueError("open_state_statuses 必须是数组")
    clean: dict[str, dict[str, Any]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("open_state_status 必须是对象")
        state_id = str(item.get("state_id", ""))
        status = str(item.get("status", "")).upper()
        if state_id not in open_ids or state_id in clean or status not in FINAL_OPEN_VALUES:
            raise ValueError("终局状态 ID/status 非法或重复")
        clean[state_id] = {
            "status": status,
            "reason": str(item.get("reason", "")).strip(),
            "confidence": _clamp_confidence(item.get("confidence")),
        }
    if set(clean) != open_ids:
        raise ValueError("终局状态未完整覆盖开放状态")
    return clean


class StateAuditFakeClient(FakeLLMClient):
    def complete(
        self,
        messages: list[dict[str, str]],
        settings: dict[str, Any],
        *,
        seed: int | None,
        purpose: str,
    ) -> str:
        del settings, seed
        if purpose.startswith("state_audit_episode"):
            return json.dumps({"new_states": [], "state_events": [], "candidate_issues": []}, ensure_ascii=False)
        if purpose.startswith("state_audit_verify"):
            prompt = messages[-1]["content"]
            match = re.search(r"候选问题：\n(.*?)\n\n当前集剧本：", prompt, re.S)
            candidates = json.loads(match.group(1)) if match else []
            return json.dumps({
                "decisions": [
                    {"candidate_id": item["candidate_id"], "verdict": "REJECTED", "evidence": "假审计拒绝", "confidence": 1.0}
                    for item in candidates
                ]
            }, ensure_ascii=False)
        if purpose.startswith("state_audit_finalize"):
            prompt = messages[-1]["content"]
            match = re.search(r"待终审开放状态：\n(.*?)\n\n输出格式：", prompt, re.S)
            states = json.loads(match.group(1)) if match else []
            return json.dumps({
                "open_state_statuses": [
                    {"state_id": item["state_id"], "status": "OPEN_VALID", "reason": "假审计保留", "confidence": 1.0}
                    for item in states
                ]
            }, ensure_ascii=False)
        return super().complete(messages, {}, seed=None, purpose=purpose)


def _apply_event(state: dict[str, Any], event: dict[str, Any], episode_id: str) -> None:
    state.setdefault("history", []).append({"episode_id": episode_id, **copy.deepcopy(event)})
    state["last_update_episode"] = episode_id
    if event["event"] in {"UPDATED", "TRANSFERRED"}:
        state["value"] = event["new_value"]
        state["description"] = _state_description(state.get("subject", "STORY"), state.get("attribute", "state"), event["new_value"])
        state["status"] = "ACTIVE"
    elif event["event"] in CLOSING_EVENTS:
        state["status"] = event["event"]
        state["closed_episode"] = episode_id


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
    config, stories, _, _, _ = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止")
    client = StateAuditFakeClient() if fake_api else OpenAICompatibleClient.from_environment("evaluation")
    settings = dict(config["evaluation"])
    settings["temperature"] = 0.0
    settings["max_output_tokens"] = max(6144, int(settings.get("max_output_tokens", 6144)))
    stories_by_id = {item["story_id"]: item for item in stories}
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
    episode_path = output_dir / "state_audit_episodes.jsonl"
    summary_path = output_dir / "state_audit_trajectories.jsonl"
    existing_episodes = index_jsonl(episode_path, "job_id")
    existing_summaries = index_jsonl(summary_path, "trajectory_id")
    evaluator_model = getattr(client, "model", "fake")
    settings_hash = stable_hash(settings)
    expected_episode_ids = list(config["episode_ids"])
    completed = 0

    for trajectory_id in trajectory_ids:
        rows = sorted(grouped[trajectory_id], key=lambda row: _episode_number(row["episode_id"]))
        if [row["episode_id"] for row in rows] != expected_episode_ids:
            raise ValueError(f"{trajectory_id} 集数不完整或顺序异常，禁止执行 E40 终局状态审计")
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
                raise ValueError(f"{trajectory_id} 已有状态审计摘要与当前输入不一致，请使用新 output-dir")
            completed += 1
            continue

        ledger = seed_initial_ledger(story)
        issues: list[dict[str, Any]] = []
        next_state_sequence = max((int(key[2:]) for key in ledger), default=0) + 1
        next_issue_sequence = 1
        resume_prefix = True

        for generation in rows:
            existing = existing_episodes.get(generation["job_id"])
            script_hash = generation.get("script_hash", stable_hash(generation["script"]))
            if existing and resume_prefix:
                if (
                    existing.get("audit_version") != AUDIT_VERSION
                    or existing.get("script_hash") != script_hash
                    or existing.get("evaluator_model") != evaluator_model
                    or existing.get("settings_hash") != settings_hash
                ):
                    raise ValueError(f"{generation['job_id']} 已有状态审计快照与当前输入不一致，请使用新 output-dir")
                ledger = {item["state_id"]: item for item in existing["states_after"]}
                issues = copy.deepcopy(existing.get("issues_after", []))
                next_state_sequence = max((int(key[2:]) for key in ledger), default=0) + 1
                next_issue_sequence = max((int(item["issue_id"][2:]) for item in issues), default=0) + 1
                continue
            if existing and not resume_prefix:
                raise ValueError(f"{generation['job_id']} 位于断点后的旧状态审计快照，禁止非连续恢复")
            resume_prefix = False

            active_states = [item for item in ledger.values() if item.get("status") == "ACTIVE"]
            active_ids = {item["state_id"] for item in active_states}
            known_ids = set(ledger)
            prompt = build_episode_audit_prompt(
                story=story,
                episode_id=generation["episode_id"],
                script=generation["script"],
                ledger_states=[ledger[key] for key in sorted(ledger)],
            )
            audit = _call_structured(
                client=client,
                prompt=prompt,
                settings=settings,
                seed=int(generation.get("seed", 0)) + 8100,
                purpose_prefix="state_audit_episode_v1",
                job=generation,
                output_dir=output_dir,
                validator=lambda value, active=active_ids, known=known_ids: validate_episode_audit(
                    value, active_ids=active, known_ids=known
                ),
            )

            for event in audit["state_events"]:
                _apply_event(ledger[event["state_id"]], event, generation["episode_id"])

            added: list[dict[str, Any]] = []
            for candidate in audit["new_states"]:
                duplicate = any(
                    item.get("status") == "ACTIVE"
                    and item.get("domain") == candidate["domain"]
                    and item.get("subject") == candidate["subject"]
                    and _overlap(item.get("description", ""), candidate["description"]) > 0.68
                    for item in ledger.values()
                )
                if duplicate:
                    continue
                state_id = _new_state_id(next_state_sequence)
                next_state_sequence += 1
                state = {
                    "state_id": state_id,
                    **copy.deepcopy(candidate),
                    "introduction_evidence": candidate["evidence"],
                    "status": "ACTIVE",
                    "introduced_episode": generation["episode_id"],
                    "last_update_episode": generation["episode_id"],
                    "closed_episode": None,
                    "history": [],
                }
                state.pop("evidence", None)
                ledger[state_id] = state
                added.append(copy.deepcopy(state))

            confirmed_this_episode: list[dict[str, Any]] = []
            candidates = audit["candidate_issues"]
            if candidates:
                candidate_ids = {item["candidate_id"] for item in candidates}
                verify_prompt = build_verification_prompt(
                    episode_id=generation["episode_id"], script=generation["script"], candidates=candidates, ledger=ledger
                )
                decisions = _call_structured(
                    client=client,
                    prompt=verify_prompt,
                    settings=settings,
                    seed=int(generation.get("seed", 0)) + 9100,
                    purpose_prefix="state_audit_verify_v1",
                    job=generation,
                    output_dir=output_dir,
                    validator=lambda value, ids=candidate_ids: validate_verification(value, ids),
                )
                for candidate in candidates:
                    decision = decisions[candidate["candidate_id"]]
                    if decision["verdict"] != "CONFIRMED":
                        continue
                    issue = {
                        "issue_id": _new_issue_id(next_issue_sequence),
                        "episode_id": generation["episode_id"],
                        **copy.deepcopy(candidate),
                        "verification": decision,
                    }
                    next_issue_sequence += 1
                    issues.append(issue)
                    confirmed_this_episode.append(copy.deepcopy(issue))

            record = {
                "audit_version": AUDIT_VERSION,
                "source_run_dir": str(run_dir),
                "evaluator_model": evaluator_model,
                "settings_hash": settings_hash,
                "job_id": generation["job_id"],
                "trajectory_id": trajectory_id,
                "story_id": generation["story_id"],
                "run_id": _job_run_id(generation),
                "condition": generation.get("condition"),
                "episode_id": generation["episode_id"],
                "script_hash": script_hash,
                "new_states": added,
                "events": audit["state_events"],
                "candidate_issues": candidates,
                "confirmed_issues": confirmed_this_episode,
                "states_after": [copy.deepcopy(ledger[key]) for key in sorted(ledger)],
                "issues_after": copy.deepcopy(issues),
                "active_count": sum(item.get("status") == "ACTIVE" for item in ledger.values()),
                "created_at": utc_now(),
            }
            append_jsonl(episode_path, record)
            existing_episodes[generation["job_id"]] = record

        open_prospective = [
            item for item in ledger.values()
            if item.get("status") == "ACTIVE" and item.get("domain") in PROSPECTIVE_DOMAINS
        ]
        final_statuses: dict[str, dict[str, Any]] = {}
        if open_prospective:
            final_job = dict(rows[-1])
            final_job["job_id"] = f"{trajectory_id}__STATE_FINAL"
            final_prompt = build_finalize_prompt(story=story, open_states=open_prospective)
            open_ids = {item["state_id"] for item in open_prospective}
            final_statuses = _call_structured(
                client=client,
                prompt=final_prompt,
                settings=settings,
                seed=int(rows[-1].get("seed", 0)) + 10100,
                purpose_prefix="state_audit_finalize_v1",
                job=final_job,
                output_dir=output_dir,
                validator=lambda value, ids=open_ids: validate_finalize(value, ids),
            )
            for state_id, status in final_statuses.items():
                ledger[state_id]["final_status"] = status["status"]
                ledger[state_id]["final_reason"] = status["reason"]
                ledger[state_id]["final_confidence"] = status["confidence"]

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
            "states": [ledger[key] for key in sorted(ledger)],
            "confirmed_issues": issues,
            "final_open_statuses": final_statuses,
            "created_at": utc_now(),
        }
        append_jsonl(summary_path, summary)
        existing_summaries[trajectory_id] = summary
        completed += 1
        print(f"[state-audit] {trajectory_id}: states={len(ledger)} issues={len(issues)}")
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
    stories_filter = {item.strip() for item in args.stories.split(",") if item.strip()} if args.stories else None
    runs_filter = {item.strip() for item in args.runs.split(",") if item.strip()} if args.runs else None
    count = audit_run(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        fake_api=args.fake_api,
        execute_api=args.execute_api,
        stories_filter=stories_filter,
        runs_filter=runs_filter,
        limit_trajectories=args.limit_trajectories,
    )
    print(f"audited_state_trajectories={count} fake_api={args.fake_api}")


if __name__ == "__main__":
    main()
