from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from .api_client import FakeLLMClient, OpenAICompatibleClient
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now, write_json
from .narrative_signature import (
    CONTROLLED_VALUES, SIGNATURE_SCHEMA_VERSION, build_signature_prompt,
    effective_count, entropy, structured_similarity, validate_signature,
)
from .run_generation import call_and_record, extract_json_value
from .schema import ROOT, load_protocol


class SignatureFakeClient(FakeLLMClient):
    def complete(self, messages: list[dict[str, str]], settings: dict[str, Any], *, seed: int | None, purpose: str) -> str:
        if purpose.startswith("narrative_signature"):
            return json.dumps({
                "macro_move": "INFO_GAIN",
                "mechanism_operations": ["INFORMATION_COMPARISON"],
                "mechanism_summary": "以两份独立信息的差异确认下一步行动对象",
                "trigger_type": "NEW_INFORMATION",
                "obstacle_type": "CONFLICTING_INFORMATION",
                "state_change_target": "FACT",
                "information_delta": "UNKNOWN_TO_KNOWN",
                "causal_relation": "CONFIRMATION",
                "turn_type": "FACT_CONFIRMED",
                "resolution_type": "NEXT_ACTION_TARGET",
                "hook_type": "PENDING_INFORMATION",
                "interaction_pattern": "COOPERATIVE_ACTION",
                "relationship_change": "NONE",
                "scene_functions": ["DISCOVERY", "VERIFICATION", "PLANNING"],
                "expressive_motifs": [],
                "evidence": "人物将新记录与既有记录对照后决定继续核查",
            }, ensure_ascii=False)
        return super().complete(messages, settings, seed=seed, purpose=purpose)


def _artifact_path(run_dir: Path, stem: str, suffix: str) -> Path:
    version = SIGNATURE_SCHEMA_VERSION.replace(".", "_")
    return run_dir / f"{stem}_v{version}{suffix}"


def _coerce_controlled(candidate: dict[str, Any]) -> dict[str, Any]:
    """将任意不在受控词表内的值统一映射为各字段预留的 OTHER 兜底值，
    避免模型偶发造词导致整批签名任务中止。OTHER 是 schema 显式允许的兜底。"""
    for field, allowed in CONTROLLED_VALUES.items():
        if field not in candidate:
            continue
        current = candidate[field]
        if isinstance(current, list):
            coerced: list[Any] = [item if item in allowed else "OTHER" for item in current]
            seen: list[Any] = []
            for item in coerced:
                if item not in seen:
                    seen.append(item)
            candidate[field] = seen
        else:
            if current not in allowed:
                candidate[field] = "OTHER"
    return candidate


def evaluate_signatures(run_dir: Path, *, fake_api: bool, execute_api: bool, limit: int | None = None, stories: set[str] | None = None, runs: set[str] | None = None) -> int:
    config, _, _, _, _ = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止")
    client = SignatureFakeClient() if fake_api else OpenAICompatibleClient.from_environment("evaluation")
    generations = read_jsonl(run_dir / "generations.jsonl")
    if stories:
        generations = [g for g in generations if g["job_id"].split("__")[0] in stories]
    if runs:
        generations = [g for g in generations if g["job_id"].split("__")[2] in runs]
    if limit is not None:
        generations = generations[:limit]
    extractions = index_jsonl(run_dir / "extractions.jsonl", "job_id")
    signature_path = _artifact_path(run_dir, "narrative_signatures", ".jsonl")
    existing = index_jsonl(signature_path, "job_id")
    completed = 0
    for generation in generations:
        job_id = generation["job_id"]
        if job_id in existing:
            completed += 1
            continue
        extraction = extractions[job_id]["result"]
        macro_move = extraction["primary_move_id"]
        prompt = build_signature_prompt(script=generation["script"], macro_move=macro_move, move_evidence=extraction.get("move_evidence", ""))
        result: dict[str, Any] | None = None
        last_error = ""
        for attempt in range(5):
            retry = "" if attempt == 0 else f"\n\n上次结构未通过：{last_error}。重新输出完整 JSON，所有分类字段只能逐字使用上方受控值，不得自造近义标签；若不确定，使用 OTHER。"
            raw = call_and_record(client, messages=[{"role": "user", "content": prompt + retry}], settings=config["evaluation"], seed=generation["seed"] + 200 + attempt, purpose=f"narrative_signature_{attempt + 1}", job=generation, output_dir=run_dir)
            try:
                candidate = extract_json_value(raw)
                candidate = _coerce_controlled(candidate)
                validate_signature(candidate, macro_move)
                result = candidate
                break
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        if result is None:
            raise ValueError(f"{job_id} 细粒度签名连续 5 次失败：{last_error}")
        record = {
            "signature_id": stable_hash({"job_id": job_id, "schema": SIGNATURE_SCHEMA_VERSION, "result": result}),
            "schema_version": SIGNATURE_SCHEMA_VERSION,
            "job_id": job_id,
            "trajectory_id": generation["trajectory_id"],
            "story_id": generation["story_id"],
            "condition": generation["condition"],
            "episode_id": generation["episode_id"],
            "signature": result,
            "created_at": utc_now(),
        }
        append_jsonl(signature_path, record)
        existing[job_id] = record
        completed += 1
    return completed


def analyze_signatures(run_dir: Path) -> dict[str, Any]:
    signature_path = _artifact_path(run_dir, "narrative_signatures", ".jsonl")
    rows = sorted(read_jsonl(signature_path), key=lambda row: (row["trajectory_id"], row["episode_id"]))
    pairs: list[dict[str, Any]] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            if left["trajectory_id"] != right["trajectory_id"]:
                continue
            metrics = structured_similarity(left["signature"], right["signature"])
            pairs.append({
                "trajectory_id": left["trajectory_id"],
                "left_episode": left["episode_id"],
                "right_episode": right["episode_id"],
                "adjacent": int(int(right["episode_id"][1:]) - int(left["episode_id"][1:]) == 1),
                **metrics,
            })
    dimensions = ["macro_move", "trigger_type", "obstacle_type", "state_change_target", "information_delta", "causal_relation", "turn_type", "resolution_type", "hook_type", "interaction_pattern", "relationship_change"]
    dimension_metrics: dict[str, Any] = {}
    for field in dimensions:
        values = [row["signature"][field] for row in rows]
        counts = {value: values.count(value) for value in sorted(set(values))}
        dimension_metrics[field] = {
            "counts": counts,
            "entropy": round(entropy(values), 6),
            "effective_count": round(effective_count(values), 6),
            "dominant_share": round(max(counts.values()) / len(values), 6) if values else 0.0,
        }
    operations = [operation for row in rows for operation in row["signature"]["mechanism_operations"]]
    dimension_metrics["mechanism_operations"] = {
        "counts": {value: operations.count(value) for value in sorted(set(operations))},
        "entropy": round(entropy(operations), 6),
        "effective_count": round(effective_count(operations), 6),
        "dominant_share": round(max(Counter(operations).values()) / len(operations), 6) if operations else 0.0,
    }
    adjacent = [pair for pair in pairs if pair["adjacent"]]
    same_macro = [pair for pair in pairs if pair["macro_move_match"]]
    summary = {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "episode_count": len(rows),
        "dimension_metrics": dimension_metrics,
        "pairwise": {
            "all_pair_mean_core_similarity": round(mean(pair["core_structural_similarity"] for pair in pairs), 6) if pairs else None,
            "adjacent_mean_core_similarity": round(mean(pair["core_structural_similarity"] for pair in adjacent), 6) if adjacent else None,
            "same_macro_mean_core_similarity": round(mean(pair["core_structural_similarity"] for pair in same_macro), 6) if same_macro else None,
            "same_macro_pair_count": len(same_macro),
            "high_repetition_pairs": [pair for pair in pairs if pair["repetition_class"] == "HIGH_REPETITION"],
            "partial_skeleton_reuse_pairs": [pair for pair in pairs if pair["repetition_class"] == "PARTIAL_SKELETON_REUSE"],
            "classification_rule": "HIGH_REPETITION requires calibrated core >= 0.75, matching information_delta, turn_type and resolution_type, plus mechanism overlap >= 0.5",
        },
        "signatures": [{"episode_id": row["episode_id"], **row["signature"]} for row in rows],
        "controlled_values": {field: sorted(values) for field, values in CONTROLLED_VALUES.items()},
    }
    write_json(_artifact_path(run_dir, "narrative_signature_summary", ".json"), summary)
    write_json(_artifact_path(run_dir, "narrative_signature_pairs", ".json"), pairs)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    parser.add_argument("--fake-api", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stories", type=str, help="逗号分隔，如 S02,S03")
    parser.add_argument("--runs", type=str, help="逗号分隔，如 R01,R02")
    parser.add_argument("--no-summary", action="store_true", help="跳过 analyze_signatures 汇总（分片提取时用，最后统一汇总）")
    args = parser.parse_args()
    stories = set(args.stories.split(",")) if args.stories else None
    runs = set(args.runs.split(",")) if args.runs else None
    count = evaluate_signatures(args.run_dir, fake_api=args.fake_api, execute_api=args.execute_api, limit=args.limit, stories=stories, runs=runs)
    summary = analyze_signatures(args.run_dir) if not args.no_summary else {"episode_count": 0}
    print(f"signature_jobs={count} episodes={summary['episode_count']}")


if __name__ == "__main__":
    main()
