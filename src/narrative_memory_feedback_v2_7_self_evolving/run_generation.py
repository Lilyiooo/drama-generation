from __future__ import annotations

import argparse
import copy
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .api_client import FakeLLMClient, LLMClient, OpenAICompatibleClient
from .build_jobs import build_jobs, write_job_bundle
from .experience_lifecycle import SelfEvolvingV4Pool
from .character_trace import apply_character_trace_lifecycle, build_character_trace_extraction_prompt, character_trace_gate_block, coerce_character_trace_extraction, generation_view, validate_character_trace_extraction
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now
from .memory import MemoryPool
from .obligations import OBLIGATION_TYPES, STRUCTURAL_OBLIGATION_TYPES, apply_obligation_lifecycle, validate_obligation_fields
from .prompts import SYSTEM_PROMPT, build_branch_prompt, build_compact_extraction_prompt, build_extraction_prompt, build_generation_prompt
from .relationship_memory import apply_relationship_lifecycle, build_relationship_extraction_prompt, coerce_relationship_extraction, generation_view as relationship_generation_view, looks_like_relationship_plan, validate_relationship_extraction
from .repair_diagnostics import detect_repair_diagnostics
from .retrieval_v4 import RetrievalDecision, load_pool as load_v4_pool, phase_for_episode, select_v4_card, select_v4_function_only, select_v4_gate_only, select_v4_gate_repair, select_v4_strategy_only, select_v4_trial_card, v4_experience_block
from .schema import ROOT, STATE_FIELDS, load_protocol, validate_state
from .state_tags import extract_state_tags


def extract_json_value(text: str, opening: str = "{", closing: str = "}") -> Any:
    del closing
    cleaned = text.strip()
    expected_type = dict if opening == "{" else list
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, Any]] = []
    for start, char in enumerate(cleaned):
        if char != opening:
            continue
        try:
            value, end = decoder.raw_decode(cleaned, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, expected_type):
            candidates.append((end - start, value))
    if not candidates:
        raise ValueError("模型结果不包含完整合法的 JSON 值")
    return max(candidates, key=lambda item: item[0])[1]


def _filter_obligations_for_condition(
    obligations: list[dict[str, Any]],
    *,
    allowed_types: set[str] | frozenset[str],
    relationship_memories: list[dict[str, Any]],
    exclude_relationship_debt: bool,
) -> list[dict[str, Any]]:
    """执行条件级义务所有权边界，覆盖新抽取、旧记录和断点恢复。"""
    filtered: list[dict[str, Any]] = []
    for item in obligations:
        if item.get("type") not in allowed_types:
            continue
        if exclude_relationship_debt and looks_like_relationship_plan(item, relationship_memories):
            continue
        filtered.append(copy.deepcopy(item))
    return filtered


def coerce_extraction_result(result: dict[str, Any], obligation_types: set[str] | frozenset[str] = OBLIGATION_TYPES) -> dict[str, Any]:
    normalized = copy.deepcopy(result)
    for wrapper in ("result", "output", "data"):
        nested = normalized.get(wrapper)
        if "state_delta" not in normalized and isinstance(nested, dict):
            normalized = copy.deepcopy(nested)
            break
    raw = normalized.get("state_delta")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            pass
    if raw == []:
        raw = {}
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict):
        raw = raw[0]
    normalized["state_delta"] = raw
    for field in ("resolved_goals", "resolved_unknown", "retracted_facts", "resolved_obligations"):
        if not isinstance(normalized.get(field), list):
            normalized[field] = []
    # new_obligations 容错：模型偶发输出非法格式，这里丢弃非法元素，避免整集 extraction 失败
    obligations = normalized.get("new_obligations")
    if not isinstance(obligations, list):
        normalized["new_obligations"] = []
    else:
        cleaned: list[dict[str, Any]] = []
        for o in obligations:
            if not isinstance(o, dict):
                continue
            item = dict(o)
            item["type"] = str(item.get("type", "")).strip().upper()
            if item["type"] not in obligation_types:
                continue  # type 非法，丢弃该义务
            if item["type"] == "DEADLINE":
                dl = item.get("deadline")
                if not isinstance(dl, str) or not dl.strip():
                    continue  # DEADLINE 缺期限，丢弃
            elif not isinstance(item.get("deadline"), str):
                item["deadline"] = None
            if not isinstance(item.get("participants"), list):
                item["participants"] = []
            if not isinstance(item.get("description"), str) or not item["description"].strip():
                continue
            if not isinstance(item.get("required_payoff"), str) or not item["required_payoff"].strip():
                continue
            cleaned.append(item)
        normalized["new_obligations"] = cleaned
    return normalized


MOVE_ID_SUFFIX_SYNONYMS = {
    "REVEAL": "GAIN",
    "REVELATION": "GAIN",
    "DISCOVERY": "GAIN",
    "DISCOVER": "GAIN",
    "UNCOVER": "GAIN",
    "HIDE": "LOSS",
    "CONCEAL": "LOSS",
    "BLOCK": "LOSS",
    "BREAK": "LOSS",
    "SHIFT": "REFRAME",
    "REDEFINE": "REFRAME",
    "REINTERPRET": "REFRAME",
    "REVISE": "REFRAME",
    "FIX": "LOCK",
    "SEAL": "LOCK",
    "FREEZE": "LOCK",
    "COMMIT": "LOCK",
}


def normalize_move_id(value: Any, move_ids: list[str]) -> Any:
    """把模型自造的近义 move_id 归一到分类法内的合法值，无法归一时原样返回以便校验拦截。"""
    if not isinstance(value, str):
        return value
    candidate = value.strip().upper().replace("-", "_").replace(" ", "_")
    if candidate in move_ids:
        return candidate
    prefix, separator, suffix = candidate.rpartition("_")
    if separator:
        mapped = f"{prefix}_{MOVE_ID_SUFFIX_SYNONYMS.get(suffix, suffix)}"
        if mapped in move_ids:
            return mapped
    return value


def _normalize_ratio_score(value: Any) -> Any:
    """把模型偶发输出的 0-10 或 0-100 分制、字符串数字统一归一到 [0,1]。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, (int, float)):
        if 1 < value <= 10:
            return value / 10
        if 10 < value <= 100:
            return value / 100
    return value


def normalize_move_fields(result: dict[str, Any], move_ids: list[str]) -> dict[str, Any]:
    result["primary_move_id"] = normalize_move_id(result.get("primary_move_id"), move_ids)
    secondary = result.get("secondary_move_ids")
    if isinstance(secondary, list):
        result["secondary_move_ids"] = [normalize_move_id(item, move_ids) for item in secondary]
    candidate = result.get("writeback_candidate")
    if isinstance(candidate, dict):
        if "primary_move_id" in candidate:
            candidate["primary_move_id"] = normalize_move_id(candidate.get("primary_move_id"), move_ids)
        if "move_id" in candidate:
            candidate["move_id"] = normalize_move_id(candidate.get("move_id"), move_ids)
        for field in ("quality_score", "confidence"):
            if field in candidate:
                candidate[field] = _normalize_ratio_score(candidate.get(field))
    return result


def normalize_delta(result: dict[str, Any]) -> dict[str, Any]:
    raw = result.get("state_delta")
    if not isinstance(raw, dict):
        raise ValueError(f"state_delta 必须是对象，实际为 {type(raw).__name__}")
    delta: dict[str, Any] = {}
    for field in STATE_FIELDS:
        value = raw.get(field, [])
        if not isinstance(value, (list, dict)):
            raise ValueError(f"state_delta.{field} 必须是数组或对象")
        delta[field] = value
    return delta


def validate_extraction_result(result: dict[str, Any], move_ids: list[str], expected_trial_ids: set[str] | None = None, obligation_types: set[str] | frozenset[str] = OBLIGATION_TYPES) -> None:
    normalize_delta(result)
    for field in ("resolved_goals", "resolved_unknown", "retracted_facts"):
        if not isinstance(result.get(field), list):
            raise ValueError(f"{field} 必须是数组")
    validate_obligation_fields(result, obligation_types)
    primary_move_id = result.get("primary_move_id")
    if primary_move_id not in move_ids:
        raise ValueError(f"primary_move_id 缺失或非法：{primary_move_id}")
    secondary_move_ids = result.get("secondary_move_ids", [])
    if not isinstance(secondary_move_ids, list) or any(move_id not in move_ids for move_id in secondary_move_ids):
        raise ValueError("secondary_move_ids 必须是合法 Move ID 数组")
    if not isinstance(result.get("state_consistency_issues", []), list):
        raise ValueError("state_consistency_issues 必须是数组")
    candidate = result.get("writeback_candidate")
    if candidate is not None:
        if not isinstance(candidate, dict):
            raise ValueError("writeback_candidate 必须是对象或 null")
        candidate_move = candidate.get("primary_move_id") or candidate.get("move_id")
        if candidate_move not in move_ids:
            raise ValueError("writeback_candidate 包含非法 move_id")
        for field in ("quality_score", "confidence"):
            value = candidate.get(field)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"writeback_candidate.{field} 必须在 [0,1]")
    feedback = result.get("card_feedback", [])
    if not isinstance(feedback, list) or any(not isinstance(item, dict) for item in feedback):
        raise ValueError("card_feedback 必须是对象数组")
    feedback_ids = [str(item.get("memory_id", "")) for item in feedback]
    expected_ids = expected_trial_ids or set()
    if set(feedback_ids) != expected_ids or len(feedback_ids) != len(set(feedback_ids)):
        raise ValueError(f"card_feedback 必须逐一覆盖受控试用卡：{sorted(expected_ids)}")
    for item in feedback:
        for field in ("adopted", "expected_effect_achieved", "state_supported", "anchor_supported", "causal_supported"):
            if not isinstance(item.get(field), bool):
                raise ValueError(f"card_feedback.{field} 必须是布尔值")
        if not isinstance(item.get("evidence", ""), str) or not isinstance(item.get("harm_flags", []), list):
            raise ValueError("card_feedback 的 evidence/harm_flags 格式非法")


def _state_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _state_strings(item)]
    if isinstance(value, dict):
        return [text for item in value.values() for text in _state_strings(item)]
    return []


def validate_state_delta_semantics(result: dict[str, Any], episode_plan: dict[str, Any]) -> None:
    delta = normalize_delta(result)
    author_markers = ("hard anchor", "hard_anchors", "实验条件", "经验卡", "本集必须", "结尾必须", "本集不得", "不得出现")
    field_markers = {
        "confirmed_facts": ("不能确认", "必须保留", "只能确认", "需要维持", "集末仍然未知"),
        "current_goals": ("保持证据边界", "维持悬念", "保留竞争解释", "避免锁定", "防止调查空间封闭", "验证实验"),
        "relationship_state": ("你查", "我查", "盯设备", "盯人", "分别分析", "独立分析", "负责核查", "调查分工"),
    }
    anchors = {anchor.strip() for anchor in episode_plan.get("hard_anchors", [])}
    for index, obligation in enumerate(result.get("new_obligations", [])):
        if not isinstance(obligation, dict):
            continue
        for field in ("description", "required_payoff"):
            text = str(obligation.get(field, "")).strip()
            marker = next((item for item in author_markers if item in text), None)
            if marker or text in anchors:
                raise ValueError(f"new_obligations[{index}].{field} 含作者规则或复制 hard anchor")
    for field, value in delta.items():
        for text in _state_strings(value):
            stripped = text.strip()
            marker = next((item for item in author_markers if item in stripped), None)
            if marker:
                raise ValueError(f"state_delta.{field} 含作者规则标记：{marker}")
            marker = next((item for item in field_markers.get(field, ()) if item in stripped), None)
            if marker:
                raise ValueError(f"state_delta.{field} 含非状态内容：{marker}")
            if stripped in anchors:
                raise ValueError(f"state_delta.{field} 直接复制 hard anchor")


def _merge(base: Any, incoming: Any) -> Any:
    if isinstance(base, list):
        additions = incoming if isinstance(incoming, list) else [incoming]
        return copy.deepcopy(base) + [copy.deepcopy(item) for item in additions if item not in base]
    if isinstance(base, dict):
        result = copy.deepcopy(base)
        source = incoming if isinstance(incoming, dict) else {"updates": incoming}
        for key, value in source.items():
            result[key] = _merge(result[key], value) if key in result else copy.deepcopy(value)
        return result
    return copy.deepcopy(incoming)


def merge_state(previous: dict[str, Any], delta: dict[str, Any]) -> dict[str, Any]:
    validate_state(previous, "previous_state")
    result = copy.deepcopy(previous)
    for field in STATE_FIELDS:
        result[field] = _merge(result[field], delta[field])
    validate_state(result, "next_state")
    return result


# ── 生命周期（extraction 直接输出 resolved 字段，merge 后做减法）──
# 实验 2/3/3.5 证明「事后推断完成」不可行（自由标注激进、2-gram 保守、结构化关联保守），
# 故 resolved 字段由 extraction 时模型直接输出（有上一状态 + 本集剧本上下文），apply_lifecycle 据此做减法。


def _overlap(a: str, b: str) -> float:
    sa = {a[i:i + 2] for i in range(max(0, len(a) - 1))}
    sb = {b[i:i + 2] for i in range(max(0, len(b) - 1))}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def apply_lifecycle(state: dict[str, Any], resolved_goals: list[str] | None = None, resolved_unknown: list[str] | None = None, retracted_facts: list[str] | None = None) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """基于 extraction 直接输出的 resolved 字段做减法（物理移除失效状态）。

    区别于「事后推断」（实验 2/3/3.5 都失败）：resolved 字段来自 extraction 时模型对
    「本集解决了什么」的直接判断（有上一状态 + 本集剧本的完整上下文）。
    匹配用「逐字引用优先 + 2-gram fallback（阈值 0.5）」。

    返回 (清理后的 state, removed 记录)。"""
    cleaned = copy.deepcopy(state)
    removed: dict[str, list[str]] = {"resolved_goals": [], "resolved_unknown": [], "retracted_facts": []}

    def _match(text: str, refs: list[str]) -> bool:
        return any(r == text or _overlap(r, text) > 0.5 for r in refs)

    # RESOLVE goals
    if resolved_goals:
        goals = cleaned.get("current_goals", [])
        keep: list[str] = []
        for g in goals:
            if _match(str(g), resolved_goals):
                removed["resolved_goals"].append(str(g))
            else:
                keep.append(g)
        cleaned["current_goals"] = keep

    # RESOLVE unknown
    if resolved_unknown:
        unknowns = cleaned.get("unknown_information", [])
        keep = []
        for u in unknowns:
            if _match(str(u), resolved_unknown):
                removed["resolved_unknown"].append(str(u))
            else:
                keep.append(u)
        cleaned["unknown_information"] = keep

    # RETRACT facts
    if retracted_facts:
        for field in ("confirmed_facts", "known_information"):
            items = cleaned.get(field, [])
            keep = []
            for x in items:
                if _match(str(x), retracted_facts):
                    removed["retracted_facts"].append(str(x))
                else:
                    keep.append(x)
            cleaned[field] = keep

    return cleaned, removed


def call_and_record(client: LLMClient, *, messages: list[dict[str, str]], settings: dict[str, Any], seed: int | None, purpose: str, job: dict[str, Any], output_dir: Path) -> str:
    call_id = f"{job['job_id']}__{purpose}"
    started_at = utc_now()
    request_hash = stable_hash({"messages": messages, "settings": settings, "seed": seed, "purpose": purpose})
    try:
        response = client.complete(messages, settings, seed=seed, purpose=purpose)
    except Exception as exc:
        append_jsonl(output_dir / "api_calls.jsonl", {"call_id": call_id, "job_id": job["job_id"], "purpose": purpose, "started_at": started_at, "finished_at": utc_now(), "request_hash": request_hash, "success": False, "error_type": type(exc).__name__})
        raise
    append_jsonl(output_dir / "api_calls.jsonl", {"call_id": call_id, "job_id": job["job_id"], "purpose": purpose, "started_at": started_at, "finished_at": utc_now(), "request_hash": request_hash, "response_hash": stable_hash(response), "success": True})
    return response


def clients_for_mode(*, fake_api: bool) -> tuple[LLMClient, LLMClient, LLMClient]:
    if fake_api:
        client = FakeLLMClient()
        return client, client, client
    return (
        OpenAICompatibleClient.from_environment("generation"),
        OpenAICompatibleClient.from_environment("extraction"),
        OpenAICompatibleClient.from_environment("evaluation"),
    )


def run(output_dir: Path, *, fake_api: bool, execute_api: bool, limit: int | None = None, conditions: set[str] | None = None, story_ids: set[str] | None = None, run_ids: set[str] | None = None) -> int:
    config, stories, plans, memory_cards, taxonomy = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止：需用户审核后同时启用 runtime.api_approved 与 --execute-api")
    if not (output_dir / "jobs.jsonl").exists():
        write_job_bundle(output_dir)
    all_jobs = read_jsonl(output_dir / "jobs.jsonl")
    if conditions:
        unknown = conditions - set(config["conditions"])
        if unknown:
            raise ValueError(f"未知条件：{sorted(unknown)}")
        all_jobs = [job for job in all_jobs if job["condition"] in conditions]
    if story_ids:
        unknown = story_ids - {s["story_id"] for s in stories}
        if unknown:
            raise ValueError(f"未知故事：{sorted(unknown)}")
        all_jobs = [job for job in all_jobs if job["story_id"] in story_ids]
    if run_ids:
        unknown = run_ids - {f"R{n:02d}" for n in range(1, config["runs_per_condition"] + 1)}
        if unknown:
            raise ValueError(f"未知 run：{sorted(unknown)}")
        all_jobs = [job for job in all_jobs if job["run_id"] in run_ids]
    selected_jobs = all_jobs[:limit] if limit is not None else all_jobs
    stories_by_id = {item["story_id"]: item for item in stories}
    plans_by_key = {(item["story_id"], item["episode_id"]): item for item in plans}
    move_ids = [item["move_id"] for item in taxonomy]
    generation_index = index_jsonl(output_dir / "generations.jsonl", "job_id")
    extraction_index = index_jsonl(output_dir / "extractions.jsonl", "job_id")
    state_index = index_jsonl(output_dir / "state_updates.jsonl", "job_id")
    obligation_index = index_jsonl(output_dir / "obligation_updates.jsonl", "job_id")
    existing_context_job_ids = {row["job_id"] for row in read_jsonl(output_dir / "contexts.jsonl")}
    existing_retrievals_by_job: dict[str, list[dict[str, Any]]] = defaultdict(list)
    existing_retrievals_by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(output_dir / "retrieval_events.jsonl"):
        existing_retrievals_by_job[row["job_id"]].append(row)
        existing_retrievals_by_trajectory[row["trajectory_id"]].append(row)
    existing_writebacks = defaultdict(list)
    for row in read_jsonl(output_dir / "writebacks.jsonl"):
        existing_writebacks[row["trajectory_id"]].append(row)
    existing_experience_updates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(output_dir / "experience_memory_updates.jsonl"):
        existing_experience_updates[row["trajectory_id"]].append(row)
    character_trace_index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_dir / "character_trace_updates.jsonl"):
        character_trace_index[row["job_id"]] = row
    relationship_index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(output_dir / "relationship_memory_updates.jsonl"):
        relationship_index[row["job_id"]] = row
    generation_client, extraction_client, evaluation_client = clients_for_mode(fake_api=fake_api)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in selected_jobs:
        grouped[job["trajectory_id"]].append(job)
    completed = 0
    v4_pool: list[dict[str, Any]] | None = None
    for trajectory_id in sorted(grouped):
        trajectory_jobs = sorted(grouped[trajectory_id], key=lambda item: item["episode_index"])
        first = trajectory_jobs[0]
        story = stories_by_id[first["story_id"]]
        condition_settings = config["conditions"][first["condition"]]
        feed_state = condition_settings.get("feed_state", True)
        feed_previous_script = condition_settings.get("feed_previous_script", False)
        enable_lifecycle = condition_settings.get("lifecycle", True)
        enable_obligations = condition_settings.get("obligation_memory", False)
        self_evolving_mode = str(condition_settings.get("self_evolving_v4", "none"))
        if self_evolving_mode not in {"none", "expert_only", "naive", "gated"}:
            raise ValueError(f"未知 self_evolving_v4 模式：{self_evolving_mode}")
        enable_v4 = bool(condition_settings.get("v4_gated_cards", False) or self_evolving_mode != "none")
        v4_lenient = bool(condition_settings.get("v4_lenient", False))
        v4_variant = condition_settings.get("v4_variant", "full")
        repair_diagnostics_mode = condition_settings.get("repair_diagnostics", "basic")
        obligation_retrieval = bool(condition_settings.get("obligation_retrieval", False))
        character_trace_mode = str(condition_settings.get("character_trace", "none"))
        if character_trace_mode not in {"none", "framing_only", "trace", "emotion_only", "agency_only"}:
            raise ValueError(f"未知 character_trace 模式：{character_trace_mode}")
        character_framing = character_trace_mode in {"framing_only", "trace", "emotion_only", "agency_only"}
        enable_character_trace = character_trace_mode in {"trace", "emotion_only", "agency_only"}
        character_trace_gate = bool(condition_settings.get("character_trace_gate", False))
        enable_relationship_memory = bool(condition_settings.get("relationship_memory", False))
        obligation_exclude_relationship_debt = bool(condition_settings.get("obligation_exclude_relationship_debt", False))
        active_obligation_types = STRUCTURAL_OBLIGATION_TYPES if obligation_exclude_relationship_debt else OBLIGATION_TYPES
        pool = MemoryPool(memory_cards, condition=first["condition"], story_id=first["story_id"])
        for row in sorted(existing_writebacks[trajectory_id], key=lambda item: item["episode_id"]):
            card = row["card"]
            pool.cards[card["memory_id"]] = card
        for event in existing_retrievals_by_trajectory[trajectory_id]:
            if "cluster_id" not in event:
                continue  # V4 门控结构 event（无 cluster_id），跳过旧 MemoryPool 曝光统计
            if event.get("selected", True):
                pool.exposures[event["cluster_id"]] += 1
        if enable_v4 and v4_pool is None:
            v4_pool = load_v4_pool()
        experience_pool: SelfEvolvingV4Pool | None = None
        recent_card_ids: list[str] = []
        cooldown_state: dict[str, int] = {}
        if self_evolving_mode != "none":
            experience_pool = SelfEvolvingV4Pool(v4_pool or [], config["experience_lifecycle"])
            snapshots = sorted(existing_experience_updates[trajectory_id], key=lambda item: item["episode_index"])
            if snapshots:
                latest = snapshots[-1]
                experience_pool.restore(latest["pool_state"])
                recent_card_ids = list(latest.get("recent_card_ids", []))
                cooldown_state = {str(key): int(value) for key, value in latest.get("cooldown_state", {}).items()}
                completed_indices = [
                    item["episode_index"] for item in trajectory_jobs if item["job_id"] in state_index
                ]
                if completed_indices and latest["episode_index"] != max(completed_indices):
                    raise RuntimeError(
                        f"{trajectory_id} 的状态与经验卡快照不一致；请从最后一个完整 episode 恢复"
                    )
            elif self_evolving_mode in {"naive", "gated"} and any(
                item["job_id"] in state_index for item in trajectory_jobs
            ):
                raise RuntimeError(f"{trajectory_id} 已有状态但缺少经验卡快照，拒绝不完整恢复")
        state = copy.deepcopy(story["initial_state"])
        # 止损机制：已 resolved 的目标/未知原文（软删除记录），供下一集检测「误删」并复活。
        resolved_pool: list[dict[str, Any]] = []
        open_obligations: list[dict[str, Any]] = []
        # 人物因果痕迹（逐轨迹隔离）：情绪痕迹 + 主体性痕迹。
        emotional_traces: list[dict[str, Any]] = []
        agency_traces: list[dict[str, Any]] = []
        # 人物关系记忆（逐轨迹隔离）：关系债独占处理，避免被结构义务 PLAN 化。
        active_relationship_memories: list[dict[str, Any]] = []
        # V4 门控经验卡状态（逐轨迹隔离）：prev_state 用于变化敏感标签，recent_moves 用于退行检测。
        prev_state: dict[str, Any] | None = None
        recent_moves: list[str] = [
            extraction_index[j["job_id"]]["result"].get("primary_move_id", "")
            for j in trajectory_jobs if j["job_id"] in extraction_index
        ][-4:]
        for job in trajectory_jobs:
            if job["job_id"] in state_index:
                prev_state = state
                state = state_index[job["job_id"]]["next_state"]
                if enable_obligations and job["job_id"] in obligation_index:
                    open_obligations = obligation_index[job["job_id"]]["open_obligations"]
                if enable_character_trace and job["job_id"] in character_trace_index:
                    snapshot = character_trace_index[job["job_id"]]
                    emotional_traces = snapshot["emotional_traces"]
                    agency_traces = snapshot["agency_traces"]
                if enable_relationship_memory and job["job_id"] in relationship_index:
                    active_relationship_memories = relationship_index[job["job_id"]]["active_relationship_memories"]
                if enable_obligations:
                    open_obligations = _filter_obligations_for_condition(
                        open_obligations,
                        allowed_types=active_obligation_types,
                        relationship_memories=active_relationship_memories,
                        exclude_relationship_debt=obligation_exclude_relationship_debt,
                    )
                completed += 1
                continue
            if job["previous_job_id"] and job["previous_job_id"] in state_index:
                state = state_index[job["previous_job_id"]]["next_state"]
            plan = plans_by_key[(job["story_id"], job["episode_id"])]
            query = " ".join([story["genre"], plan["episode_goal"], *plan["open_decisions"]])
            cards: list[dict[str, Any]] = []
            retrievals = existing_retrievals_by_job[job["job_id"]]
            experience_text: str | None = None
            v4_decision = None
            diagnostic_tags: set[str] = set()
            diagnostic_evidence: dict[str, Any] = {}
            experience_context: dict[str, Any] = {}
            trial_cards_for_feedback: list[dict[str, Any]] = []
            if enable_v4:
                active_v4_cards = experience_pool.searchable_cards() if experience_pool else list(v4_pool or [])
                v4_job = dict(job)
                v4_job["hard_anchors"] = plan.get("hard_anchors", [])
                v4_job["episode_goal"] = plan.get("episode_goal", "")
                if job.get("episode_function"):
                    v4_job["episode_function"] = job["episode_function"]
                state_tags, trajectory_declining = extract_state_tags(v4_job, state, recent_moves, prev_state)
                if v4_variant == "function_only":
                    v4_decision = select_v4_function_only(v4_job, total_episodes=len(config["episode_ids"]))
                elif v4_variant == "gate_only":
                    v4_decision = select_v4_gate_only(active_v4_cards, v4_job, total_episodes=len(config["episode_ids"]))
                elif v4_variant == "gate_repair":
                    diagnostic_tags = state_tags & {"recent_move_repeat", "state_stagnation", "external_rescue_risk"}
                    diagnostic_evidence = {tag: {"source": "state_tags"} for tag in diagnostic_tags}
                    if repair_diagnostics_mode == "full":
                        prior_scripts = [
                            generation_index[prior["job_id"]]["script"]
                            for prior in trajectory_jobs
                            if prior["episode_index"] < job["episode_index"] and prior["job_id"] in generation_index
                        ][-2:]
                        previous_extraction = extraction_index.get(job.get("previous_job_id", ""), {}).get("result")
                        full_diagnostics = detect_repair_diagnostics(
                            job=v4_job,
                            plan=plan,
                            state=state,
                            recent_scripts=prior_scripts,
                            previous_extraction=previous_extraction,
                            total_episodes=len(config["episode_ids"]),
                        )
                        diagnostic_tags |= full_diagnostics.tags
                        diagnostic_evidence.update(full_diagnostics.evidence)
                    v4_decision = select_v4_gate_repair(
                        active_v4_cards, v4_job, state_tags, diagnostic_tags,
                        recent_card_ids=recent_card_ids,
                        cooldown_state=cooldown_state,
                        total_episodes=len(config["episode_ids"]),
                    )
                elif v4_variant == "strategy_only":
                    v4_decision = select_v4_strategy_only(
                        active_v4_cards, v4_job, state_tags,
                        trajectory_declining=trajectory_declining,
                        recent_card_ids=recent_card_ids,
                        cooldown_state=cooldown_state,
                        total_episodes=len(config["episode_ids"]),
                    )
                else:
                    v4_decision = select_v4_card(
                        active_v4_cards, v4_job, state_tags,
                        trajectory_declining=trajectory_declining,
                        recent_card_ids=recent_card_ids,
                        cooldown_state=cooldown_state,
                        total_episodes=len(config["episode_ids"]),
                        open_obligations=open_obligations if obligation_retrieval else None,
                    )
                episode_phase = phase_for_episode(job["episode_index"], len(config["episode_ids"]))
                formal_selected_card = v4_decision.selected_card
                experience_context = {
                    "episode_function": v4_decision.episode_function,
                    "phase": episode_phase,
                    "state_tags": sorted(state_tags),
                    "protected_invariants": v4_decision.protected_invariants,
                    "candidate_generation_allowed": (
                        formal_selected_card is None
                        or formal_selected_card.get("card_type") == "STRATEGY"
                    ),
                }
                if (
                    experience_pool and self_evolving_mode == "gated" and v4_variant == "full"
                    and (formal_selected_card is None or formal_selected_card.get("card_type") == "STRATEGY")
                ):
                    trial_decision = select_v4_trial_card(
                        experience_pool.trial_cards(), v4_job, state_tags,
                        recent_card_ids=recent_card_ids,
                        cooldown_state=cooldown_state,
                        total_episodes=len(config["episode_ids"]),
                    )
                    if trial_decision.selected_card and experience_pool.should_trial(rng_seed=job["seed"]):
                        trial_card = trial_decision.selected_card
                        v4_decision = RetrievalDecision(
                            selected_card=trial_card,
                            active_blockers=v4_decision.active_blockers,
                            episode_function=v4_decision.episode_function,
                            function_evidence=v4_decision.function_evidence,
                            protected_invariants=v4_decision.protected_invariants,
                            reason="controlled_candidate_trial",
                            top_score=trial_decision.top_score,
                            eligible_count=trial_decision.eligible_count,
                        )
                        trial_cards_for_feedback = [trial_card]
                        exposure_event = experience_pool.record_exposure(
                            trial_card["memory_id"], episode_id=job["episode_id"],
                            context_signature=stable_hash(experience_context),
                        )
                        if exposure_event:
                            append_jsonl(output_dir / "experience_card_events.jsonl", {
                                **exposure_event, "created_at": utc_now(), "job_id": job["job_id"],
                                "trajectory_id": trajectory_id, "story_id": job["story_id"],
                                "episode_id": job["episode_id"], "condition": job["condition"],
                            })
                cards = [v4_decision.selected_card] if v4_decision.selected_card else []
                experience_text = v4_experience_block(
                    v4_decision, lenient=v4_lenient, max_blockers=(1 if v4_lenient else None),
                    include_steps=not bool(trial_cards_for_feedback),
                )
                if not existing_retrievals_by_job[job["job_id"]]:
                    append_jsonl(output_dir / "retrieval_events.jsonl", {
                        "created_at": utc_now(), "job_id": job["job_id"], "trajectory_id": trajectory_id,
                        "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"],
                        "run_id": job["run_id"], "selected": bool(v4_decision.selected_card),
                        "memory_id": v4_decision.selected_card["memory_id"] if v4_decision.selected_card else None,
                        "card_type": v4_decision.selected_card["card_type"] if v4_decision.selected_card else None,
                        "mechanism_family": v4_decision.selected_card["mechanism_family"] if v4_decision.selected_card else None,
                        "card_source_type": v4_decision.selected_card.get("source_type") if v4_decision.selected_card else None,
                        "card_lifecycle_status": v4_decision.selected_card.get("lifecycle_status") if v4_decision.selected_card else None,
                        "candidate_trial": bool(trial_cards_for_feedback),
                        "episode_function": v4_decision.episode_function,
                        "protected_invariants": v4_decision.protected_invariants,
                        "active_blockers": [b["memory_id"] for b in v4_decision.active_blockers],
                        "reason": v4_decision.reason,
                        "state_tags": sorted(state_tags),
                        "diagnostic_tags": sorted(diagnostic_tags),
                        "diagnostic_evidence": diagnostic_evidence,
                    })
                    existing_retrievals_by_job[job["job_id"]].append({"selected": bool(v4_decision.selected_card), "memory_id": v4_decision.selected_card["memory_id"] if v4_decision.selected_card else None})
            elif retrievals:
                selected_retrievals = [event for event in retrievals if event.get("selected", True)]
                cards = [copy.deepcopy(pool.cards[event["memory_id"]]) for event in sorted(selected_retrievals, key=lambda item: item["rank"])]
            elif condition_settings["retrieval_enabled"]:
                base_settings = dict(config["retrieval"])
                overrides = config.get("retrieval_overrides", {}).get(job["condition"], {})
                retrieval_settings = {**base_settings, **overrides}
                cards, retrievals = pool.retrieve(query, settings=retrieval_settings, rng_seed=job["seed"])
                for event in retrievals:
                    record = {**event, "job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "run_id": job["run_id"], "condition": job["condition"], "created_at": utc_now()}
                    append_jsonl(output_dir / "retrieval_events.jsonl", record)
                    existing_retrievals_by_job[job["job_id"]].append(record)
            prev_script = None
            if feed_previous_script and job.get("previous_job_id"):
                prev_script = generation_index.get(job["previous_job_id"], {}).get("script")
            character_trace_view = generation_view(emotional_traces, agency_traces) if enable_character_trace else None
            prompt = build_generation_prompt(
                story=story, plan=plan,
                state=state if feed_state else None,
                cards=cards, output_characters=config["output_characters"],
                previous_script=prev_script,
                obligations=open_obligations if enable_obligations else None,
                relationship_memories=relationship_generation_view(active_relationship_memories) if enable_relationship_memory else None,
                experience_text=experience_text,
                character_traces=character_trace_view,
                character_framing=character_framing,
                character_trace_gate=character_trace_gate_block(character_trace_view) if (enable_character_trace and character_trace_gate) else None,
            )
            if job["job_id"] not in existing_context_job_ids:
                append_jsonl(output_dir / "contexts.jsonl", {"job_id": job["job_id"], "trajectory_id": trajectory_id, "state_hash": stable_hash(state), "retrieved_memory_ids": [card["memory_id"] for card in cards], "retrieved_cluster_ids": [card.get("cluster_id", card.get("mechanism_family", "")) for card in cards], "prompt_hash": stable_hash(prompt), "prompt": prompt if config["runtime"]["store_prompts"] else None, "created_at": utc_now()})
                existing_context_job_ids.add(job["job_id"])
            if job["job_id"] in generation_index:
                script = generation_index[job["job_id"]]["script"]
            else:
                script = call_and_record(generation_client, messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}], settings=config["generation"], seed=job["seed"], purpose="generation", job=job, output_dir=output_dir)
                generation_record = {**job, "script": script, "script_hash": stable_hash(script), "created_at": utc_now()}
                append_jsonl(output_dir / "generations.jsonl", generation_record)
                generation_index[job["job_id"]] = generation_record
            if job["job_id"] in extraction_index:
                # 旧 extraction 也必须经过当前条件的 schema 清洗；不能因为断点恢复而绕过 Q1 边界。
                extraction = normalize_move_fields(
                    coerce_extraction_result(extraction_index[job["job_id"]]["result"], active_obligation_types),
                    move_ids,
                )
                validate_extraction_result(
                    extraction,
                    move_ids,
                    expected_trial_ids={card["memory_id"] for card in trial_cards_for_feedback},
                    obligation_types=active_obligation_types,
                )
            else:
                extraction_prompt = build_extraction_prompt(
                    story=story, plan=plan, previous_state=state, script=script, move_ids=move_ids,
                    open_obligations=open_obligations if enable_obligations else None,
                    trial_cards=trial_cards_for_feedback,
                    experience_context=experience_context,
                    exclude_relationship_debt=obligation_exclude_relationship_debt,
                )
                extraction: dict[str, Any] | None = None
                last_error = ""
                previous_raw = ""
                for attempt in range(5):
                    if attempt == 0:
                        request_prompt = extraction_prompt
                    else:
                        request_prompt = build_compact_extraction_prompt(
                            script=script,
                            move_ids=move_ids,
                            validation_error=last_error,
                            trial_cards=trial_cards_for_feedback,
                            experience_context=experience_context,
                        )
                    purpose = "extraction" if attempt == 0 else f"extraction_retry_{attempt}"
                    previous_raw = call_and_record(extraction_client, messages=[{"role": "user", "content": request_prompt}], settings=config["extraction"], seed=job["seed"] + attempt, purpose=purpose, job=job, output_dir=output_dir)
                    try:
                        candidate_result = normalize_move_fields(coerce_extraction_result(extract_json_value(previous_raw), active_obligation_types), move_ids)
                        validate_extraction_result(
                            candidate_result, move_ids,
                            expected_trial_ids={card["memory_id"] for card in trial_cards_for_feedback},
                            obligation_types=active_obligation_types,
                        )
                        validate_state_delta_semantics(candidate_result, plan)
                        extraction = candidate_result
                        break
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                        last_error = str(exc)
                        append_jsonl(output_dir / "extraction_failures.jsonl", {"job_id": job["job_id"], "trajectory_id": trajectory_id, "episode_id": job["episode_id"], "attempt": attempt + 1, "error": last_error, "response_hash": stable_hash(previous_raw), "created_at": utc_now()})
                if extraction is None:
                    append_jsonl(output_dir / "skipped_jobs.jsonl", {"job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "reason": f"结构抽取连续 5 次未通过校验：{last_error}", "created_at": utc_now()})
                    # 当前集没有状态/义务/关系记忆快照，继续会让后续集在断裂状态上生成；终止本轨迹等待恢复。
                    break
                extraction_record = {"job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "result": extraction, "created_at": utc_now()}
                append_jsonl(output_dir / "extractions.jsonl", extraction_record)
                extraction_index[job["job_id"]] = extraction_record
            delta = normalize_delta(extraction)
            next_state = merge_state(state, delta)
            lifecycle_removed = {"resolved_goals": [], "resolved_unknown": [], "retracted_facts": []}
            misdeleted: list[dict[str, Any]] = []
            if enable_lifecycle:
                # 止损机制：误删检测——本集 delta 新增的目标若与 resolved_pool 里被删目标高度相似，
                # 说明上一集把仍在推进的目标误删了（本集又把它重新提出），记录并复活。
                new_goals = [str(g) for g in delta.get("current_goals", [])]
                misdeleted = [item for item in resolved_pool
                              if item["type"] == "goal" and any(_overlap(item["text"], g) > 0.6 for g in new_goals)]
                for item in misdeleted:
                    resolved_pool.remove(item)
                next_state, lifecycle_removed = apply_lifecycle(
                    next_state,
                    resolved_goals=extraction.get("resolved_goals", []),
                    resolved_unknown=extraction.get("resolved_unknown", []),
                    retracted_facts=extraction.get("retracted_facts", []),
                )
                # 软删除记录：把本集被 resolved 的目标/未知原文加入 resolved_pool，供后续集误删检测
                for g in lifecycle_removed["resolved_goals"]:
                    resolved_pool.append({"text": g, "type": "goal", "episode": job["episode_id"]})
                for u in lifecycle_removed["resolved_unknown"]:
                    resolved_pool.append({"text": u, "type": "unknown", "episode": job["episode_id"]})
            if enable_relationship_memory and job["job_id"] in relationship_index:
                active_relationship_memories = relationship_index[job["job_id"]]["active_relationship_memories"]
            elif enable_relationship_memory:
                relationship_prompt = build_relationship_extraction_prompt(
                    story=story,
                    episode_id=job["episode_id"],
                    script=script,
                    active_memories=active_relationship_memories,
                )
                relationship_extraction: dict[str, Any] | None = None
                relationship_error = ""
                for relationship_attempt in range(4):
                    retry_suffix = "" if relationship_attempt == 0 else f"\n\n上次输出未通过校验：{relationship_error}。重新输出完整合法 JSON。"
                    raw_relationship = call_and_record(
                        extraction_client,
                        messages=[{"role": "user", "content": relationship_prompt + retry_suffix}],
                        settings=config["extraction"],
                        seed=job["seed"] + 200 + relationship_attempt,
                        purpose=f"relationship_extraction_{relationship_attempt + 1}",
                        job=job,
                        output_dir=output_dir,
                    )
                    try:
                        relationship_extraction = validate_relationship_extraction(
                            coerce_relationship_extraction(extract_json_value(raw_relationship)),
                            active_ids={item["relationship_id"] for item in active_relationship_memories},
                        )
                        break
                    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                        relationship_error = str(exc)
                        append_jsonl(output_dir / "relationship_extraction_failures.jsonl", {
                            "job_id": job["job_id"], "trajectory_id": trajectory_id,
                            "episode_id": job["episode_id"], "attempt": relationship_attempt + 1,
                            "error": relationship_error, "response_hash": stable_hash(raw_relationship),
                            "created_at": utc_now(),
                        })
                if relationship_extraction is None:
                    # 关系记忆是辅助记忆：抽取失败时降级为「本集不更新关系记忆」，保持上一集状态，不终止轨迹。
                    relationship_record = {
                        "job_id": job["job_id"], "trajectory_id": trajectory_id,
                        "story_id": job["story_id"], "episode_id": job["episode_id"],
                        "condition": job["condition"], "added": [], "changed": [], "resolved": [],
                        "degraded": True, "error": relationship_error,
                        "active_relationship_memories": active_relationship_memories,
                        "created_at": utc_now(),
                    }
                else:
                    active_relationship_memories, relationship_added, relationship_changed, relationship_resolved = apply_relationship_lifecycle(
                        active_relationship_memories,
                        new_memories=relationship_extraction["new_relationship_memories"],
                        updates=relationship_extraction["relationship_updates"],
                        job_id=job["job_id"],
                        episode_id=job["episode_id"],
                    )
                    relationship_record = {
                        "job_id": job["job_id"], "trajectory_id": trajectory_id,
                        "story_id": job["story_id"], "episode_id": job["episode_id"],
                        "condition": job["condition"], "added": relationship_added,
                        "changed": relationship_changed, "resolved": relationship_resolved,
                        "active_relationship_memories": active_relationship_memories,
                        "created_at": utc_now(),
                    }
                append_jsonl(output_dir / "relationship_memory_updates.jsonl", relationship_record)
                relationship_index[job["job_id"]] = relationship_record
            if enable_obligations and job["job_id"] in obligation_index:
                # 义务日志可能已写而 state 尚未写；恢复已有快照，避免同 job 重放并产生重复 JSONL 键。
                obligation_record = obligation_index[job["job_id"]]
                open_obligations = _filter_obligations_for_condition(
                    obligation_record["open_obligations"],
                    allowed_types=active_obligation_types,
                    relationship_memories=active_relationship_memories,
                    exclude_relationship_debt=obligation_exclude_relationship_debt,
                )
            elif enable_obligations:
                # 关系记忆已完成本集更新后，再做确定性分流：显式关系债和伪装成 PLAN 的关系债都不得进入结构义务池。
                open_obligations = _filter_obligations_for_condition(
                    open_obligations,
                    allowed_types=active_obligation_types,
                    relationship_memories=active_relationship_memories,
                    exclude_relationship_debt=obligation_exclude_relationship_debt,
                )
                new_obligations = _filter_obligations_for_condition(
                    extraction.get("new_obligations", []),
                    allowed_types=active_obligation_types,
                    relationship_memories=active_relationship_memories,
                    exclude_relationship_debt=obligation_exclude_relationship_debt,
                )
                open_obligations, obligations_added, obligations_resolved = apply_obligation_lifecycle(
                    open_obligations,
                    new_obligations=new_obligations,
                    resolved_descriptions=extraction.get("resolved_obligations", []),
                    job_id=job["job_id"],
                    episode_id=job["episode_id"],
                    allowed_types=active_obligation_types,
                )
                obligation_record = {
                    "job_id": job["job_id"], "trajectory_id": trajectory_id,
                    "story_id": job["story_id"], "episode_id": job["episode_id"],
                    "condition": job["condition"], "added": obligations_added,
                    "resolved": obligations_resolved,
                    "open_obligations": open_obligations,
                    "created_at": utc_now(),
                }
                append_jsonl(output_dir / "obligation_updates.jsonl", obligation_record)
                obligation_index[job["job_id"]] = obligation_record
            if enable_character_trace:
                active_emo_ids = {item["trace_id"] for item in emotional_traces}
                active_agy_ids = {item["trace_id"] for item in agency_traces}
                trace_extract_mode = {"trace": "both", "emotion_only": "emotion", "agency_only": "agency"}.get(character_trace_mode, "both")
                trace_prompt = build_character_trace_extraction_prompt(
                    story=story, episode_id=job["episode_id"], script=script,
                    emotional=emotional_traces, agency=agency_traces,
                    mode=trace_extract_mode,
                )
                trace_value: dict[str, Any] | None = None
                for trace_attempt in range(3):
                    trace_purpose = "character_trace" if trace_attempt == 0 else f"character_trace_retry_{trace_attempt}"
                    trace_raw = call_and_record(extraction_client, messages=[{"role": "user", "content": trace_prompt}], settings=config["extraction"], seed=job["seed"] + 100 + trace_attempt, purpose=trace_purpose, job=job, output_dir=output_dir)
                    try:
                        trace_value = validate_character_trace_extraction(
                            coerce_character_trace_extraction(extract_json_value(trace_raw)),
                            active_emotional_ids=active_emo_ids,
                            active_agency_ids=active_agy_ids,
                        )
                        break
                    except Exception:
                        continue
                if trace_value is not None:
                    emotional_traces, agency_traces, trace_effect = apply_character_trace_lifecycle(
                        emotional_traces, agency_traces,
                        new_emotional=trace_value["new_emotional_traces"],
                        emotional_updates=trace_value["emotional_updates"],
                        new_agency=trace_value["new_agency_traces"],
                        agency_updates=trace_value["agency_updates"],
                        job_id=job["job_id"], episode_id=job["episode_id"],
                    )
                    trace_record = {
                        "job_id": job["job_id"], "trajectory_id": trajectory_id,
                        "story_id": job["story_id"], "episode_id": job["episode_id"],
                        "condition": job["condition"], "run_id": job["run_id"],
                        "emotional_traces": emotional_traces, "agency_traces": agency_traces,
                        "effect": trace_effect, "created_at": utc_now(),
                    }
                    append_jsonl(output_dir / "character_trace_updates.jsonl", trace_record)
                    character_trace_index[job["job_id"]] = trace_record
            state_record = {"job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "previous_state_hash": stable_hash(state), "state_delta": delta, "next_state": next_state, "next_state_hash": stable_hash(next_state), "state_consistency_issues": extraction.get("state_consistency_issues", []), "lifecycle_removed": lifecycle_removed, "lifecycle_misdeleted": [item["text"] for item in misdeleted], "created_at": utc_now()}
            append_jsonl(output_dir / "state_updates.jsonl", state_record)
            state_index[job["job_id"]] = state_record
            if condition_settings["pool_mode"] == "append_all":
                candidate = extraction.get("writeback_candidate")
                if isinstance(candidate, dict) and float(candidate.get("confidence", 0)) >= config["writeback"]["minimum_confidence"] and candidate.get("move_id") in move_ids:
                    card = pool.append_writeback(job_id=job["job_id"], episode_id=job["episode_id"], candidate=candidate, parent_memory_ids=[item["memory_id"] for item in cards])
                    append_jsonl(output_dir / "writebacks.jsonl", {"writeback_id": card["memory_id"], "job_id": job["job_id"], "trajectory_id": trajectory_id, "episode_id": job["episode_id"], "condition": job["condition"], "card": card, "created_at": utc_now()})
            if experience_pool and self_evolving_mode in {"naive", "gated"}:
                context_signature = stable_hash(experience_context)
                if self_evolving_mode == "gated":
                    feedback_by_id = {
                        str(item.get("memory_id")): item
                        for item in extraction.get("card_feedback", [])
                        if isinstance(item, dict)
                    }
                    for trial_card in trial_cards_for_feedback:
                        memory_id = trial_card["memory_id"]
                        feedback = feedback_by_id[memory_id]
                        evidence = str(feedback.get("evidence", "")).strip()
                        feedback_event = experience_pool.record_feedback(
                            memory_id,
                            feedback=feedback,
                            episode_id=job["episode_id"],
                            context_signature=context_signature,
                            experience_context=experience_context,
                            evidence_valid=bool(evidence and evidence in script),
                        )
                        if feedback_event:
                            append_jsonl(output_dir / "experience_card_events.jsonl", {
                                **feedback_event, "created_at": utc_now(), "job_id": job["job_id"],
                                "trajectory_id": trajectory_id, "story_id": job["story_id"],
                                "episode_id": job["episode_id"], "condition": job["condition"],
                            })
                    for promoted_card in experience_pool.promote_ready():
                        append_jsonl(output_dir / "experience_card_events.jsonl", {
                            "action": "candidate_promoted", "memory_id": promoted_card["memory_id"],
                            "card": promoted_card, "created_at": utc_now(), "job_id": job["job_id"],
                            "trajectory_id": trajectory_id, "story_id": job["story_id"],
                            "episode_id": job["episode_id"], "condition": job["condition"],
                        })
                candidate = extraction.get("writeback_candidate")
                if isinstance(candidate, dict):
                    if not experience_context.get("candidate_generation_allowed", False):
                        candidate_event = {"action": "rejected", "reason": "existing_card_or_trial_covered_episode", "card": None}
                    elif extraction.get("state_consistency_issues"):
                        candidate_event = {"action": "rejected", "reason": "state_consistency_issue", "card": None}
                    else:
                        candidate_event = experience_pool.add_candidate(
                            candidate=candidate,
                            job_id=job["job_id"], episode_id=job["episode_id"],
                            context_signature=context_signature,
                            state_tags=set(experience_context.get("state_tags", [])),
                            episode_function=str(experience_context.get("episode_function", "PROGRESS")),
                            phase=str(experience_context.get("phase", "expansion")),
                            protected_invariants=list(experience_context.get("protected_invariants", [])),
                            parent_memory_ids=[item["memory_id"] for item in cards],
                            story_terms={story["title"], *[item["name"] for item in story["characters"]]},
                            mode=self_evolving_mode,
                        )
                    append_jsonl(output_dir / "experience_card_events.jsonl", {
                        **candidate_event, "created_at": utc_now(), "job_id": job["job_id"],
                        "trajectory_id": trajectory_id, "story_id": job["story_id"],
                        "episode_id": job["episode_id"], "condition": job["condition"],
                    })
                    accepted_card = candidate_event.get("card")
                    if accepted_card and candidate_event["action"] in {"candidate_created", "naive_promoted"}:
                        append_jsonl(output_dir / "writebacks.jsonl", {
                            "writeback_id": accepted_card["memory_id"], "job_id": job["job_id"],
                            "trajectory_id": trajectory_id, "episode_id": job["episode_id"],
                            "condition": job["condition"], "lifecycle_action": candidate_event["action"],
                            "card": accepted_card, "created_at": utc_now(),
                        })
            if config["branch_sampling"]["enabled"] and job["episode_id"] in config["branch_sampling"]["checkpoints"]:
                try:
                    branch_prompt = build_branch_prompt(story=story, plan=plan, state=next_state, count=config["branch_sampling"]["candidates_per_checkpoint"], move_ids=move_ids)
                    raw = call_and_record(evaluation_client, messages=[{"role": "user", "content": branch_prompt}], settings=config["extraction"], seed=job["seed"], purpose="branching", job=job, output_dir=output_dir)
                    branches = extract_json_value(raw, "[", "]")
                    expected_count = config["branch_sampling"]["candidates_per_checkpoint"]
                    if not isinstance(branches, list) or len(branches) != expected_count:
                        raise ValueError(f"候选分支必须严格返回 {expected_count} 项")
                    if any(not isinstance(branch, dict) or branch.get("move_id") not in move_ids for branch in branches):
                        raise ValueError("候选分支包含非法结构或未知 move_id")
                    append_jsonl(output_dir / "branches.jsonl", {"job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "branches": branches, "created_at": utc_now()})
                except Exception as exc:
                    append_jsonl(output_dir / "branch_failures.jsonl", {"job_id": job["job_id"], "trajectory_id": trajectory_id, "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "error": str(exc), "created_at": utc_now()})
            if enable_v4 and v4_decision is not None and v4_decision.selected_card:
                recent_card_ids.append(v4_decision.selected_card["memory_id"])
                cooldown_state[v4_decision.selected_card["memory_id"]] = job["episode_index"]
            recent_moves.append(extraction.get("primary_move_id", ""))
            if experience_pool and self_evolving_mode in {"naive", "gated"}:
                append_jsonl(output_dir / "experience_memory_updates.jsonl", {
                    "job_id": job["job_id"], "trajectory_id": trajectory_id,
                    "story_id": job["story_id"], "condition": job["condition"],
                    "run_id": job["run_id"], "episode_id": job["episode_id"],
                    "episode_index": job["episode_index"], "pool_state": experience_pool.snapshot(),
                    "recent_card_ids": recent_card_ids[-8:], "cooldown_state": cooldown_state,
                    "created_at": utc_now(),
                })
            prev_state = state
            state = next_state
            completed += 1
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-api", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--conditions", type=str, help="逗号分隔的条件名，仅运行这些条件（如 A_state_only,C_high_homogeneous_frozen,D_high_diverse_append）")
    parser.add_argument("--stories", type=str, help="逗号分隔的故事名，仅运行这些故事（如 S03,S04）；用于单故事抽查")
    parser.add_argument("--runs", type=str, help="逗号分隔的 run 名，仅运行这些 run（如 R01 或 R01,R02）；用于跨故事单 run 快照")
    args = parser.parse_args()
    conditions = {item.strip() for item in args.conditions.split(",") if item.strip()} if args.conditions else None
    story_ids = {item.strip() for item in args.stories.split(",") if item.strip()} if args.stories else None
    run_ids = {item.strip() for item in args.runs.split(",") if item.strip()} if args.runs else None
    config, stories_cfg, _, _, _ = load_protocol()
    planned = build_jobs(config, stories_cfg)
    if conditions:
        planned = [job for job in planned if job["condition"] in conditions]
    if story_ids:
        planned = [job for job in planned if job["story_id"] in story_ids]
    if run_ids:
        planned = [job for job in planned if job["run_id"] in run_ids]
    if args.dry_run:
        print(f"validated_jobs={len(planned)} api_calls=0 output_writes=0")
        return
    count = run(args.output_dir, fake_api=args.fake_api, execute_api=args.execute_api, limit=args.limit, conditions=conditions, story_ids=story_ids, run_ids=run_ids)
    print(f"completed_jobs={count} fake_api={args.fake_api}")


if __name__ == "__main__":
    main()
