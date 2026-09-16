from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .api_client import FakeLLMClient, OpenAICompatibleClient
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now
from .prompts import build_evaluation_prompt
from .run_generation import call_and_record, extract_json_value
from .schema import ROOT, load_protocol

SCORE_FIELDS = ("dialogue_quality", "scene_structure", "visual_adaptability", "state_consistency", "anchor_satisfaction", "causal_continuity", "mechanism_novelty", "rule_echo_avoidance")


def reconcile_scores(value: dict[str, Any]) -> None:
    """Recompute each dimension score as clamp(100 + sum(points)) from details when items exist."""
    details = value.get("details")
    if not isinstance(details, dict):
        return
    for field in SCORE_FIELDS:
        items = details.get(field)
        if not isinstance(items, list) or not items:
            continue
        points: list[float] = []
        for item in items:
            point = item.get("points") if isinstance(item, dict) else None
            if not isinstance(point, (int, float)):
                points = []
                break
            points.append(float(point))
        if points:
            value[field] = round(max(0.0, min(100.0, 100.0 + sum(points))), 2)


import re


def _coerce_numeric(value: Any) -> float | None:
    """Turn a numeric value (or a numeric-looking string like '78分'/'85%'/'90/100') into float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def reconcile_and_validate(value: dict[str, Any]) -> None:
    """Coerce scores, recompute from details, and validate; raises ValueError on unrecoverable output."""
    if not isinstance(value.get("details"), dict):
        value["details"] = {}
    if not isinstance(value.get("justifications"), dict):
        value["justifications"] = {}
    if not isinstance(value.get("violations"), list):
        value["violations"] = []
    for field in SCORE_FIELDS:
        coerced = _coerce_numeric(value.get(field))
        if coerced is not None:
            value[field] = max(0.0, min(100.0, coerced))
    reconcile_scores(value)
    for field in SCORE_FIELDS:
        score = _coerce_numeric(value.get(field))
        if score is None or not 0 <= score <= 100:
            raise ValueError(f"{field} 必须是 0-100 的数值")
        value[field] = score


def run(output_dir: Path, *, fake_api: bool, execute_api: bool, limit: int | None = None, story_filter: set[str] | None = None, run_filter: set[str] | None = None) -> int:
    config, stories, plans, _, _ = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止：需用户审核后同时启用 runtime.api_approved 与 --execute-api")
    client = FakeLLMClient() if fake_api else OpenAICompatibleClient.from_environment("evaluation")
    stories_by_id = {item["story_id"]: item for item in stories}
    plans_by_key = {(item["story_id"], item["episode_id"]): item for item in plans}
    jobs = index_jsonl(output_dir / "jobs.jsonl", "job_id")
    generations = read_jsonl(output_dir / "generations.jsonl")
    if story_filter:
        generations = [g for g in generations if g["job_id"].split("__")[0] in story_filter]
    if run_filter:
        generations = [g for g in generations if g["job_id"].split("__")[2] in run_filter]
    if limit is not None:
        generations = generations[:limit]
    states = index_jsonl(output_dir / "state_updates.jsonl", "job_id")
    existing = index_jsonl(output_dir / "evaluations.jsonl", "job_id")
    error_log = output_dir / "evaluation_errors.log"
    completed = 0
    skipped = 0
    for generation in generations:
        job = jobs[generation["job_id"]]
        if job["job_id"] in existing:
            completed += 1
            continue
        story = stories_by_id[job["story_id"]]
        plan = plans_by_key.get((job["story_id"], job["episode_id"])) or {
            "story_id": job["story_id"], "episode_id": job["episode_id"],
            "episode_goal": job.get("episode_goal", ""),
            "open_decisions": job.get("open_decisions", []),
            "hard_anchors": job.get("hard_anchors", []),
        }
        previous_state = story["initial_state"] if job["previous_job_id"] is None else states[job["previous_job_id"]]["next_state"]
        prompt = build_evaluation_prompt(story=story, plan=plan, previous_state=previous_state, script=generation["script"])
        result: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                retry_note = ""
                if attempt > 0:
                    retry_note = f"\n\n[上一次输出非法：{last_error}。必须严格输出 JSON，8 个分数为 0-100 的数值（不要带“分/百分制/%/100”等单位，不要写字符串），details 必须列出全部 8 个维度。]"
                raw = call_and_record(client, messages=[{"role": "user", "content": prompt + retry_note}], settings=config["evaluation"], seed=job["seed"], purpose="evaluation", job=job, output_dir=output_dir)
                result = extract_json_value(raw)
                if not isinstance(result, dict):
                    raise ValueError("返回结果不是 JSON 对象")
                reconcile_and_validate(result)
                break
            except Exception as exc:  # noqa: BLE001 - 单条评估失败需重试或跳过，不能中断全量
                last_error = exc
                result = None
        if result is None:
            skipped += 1
            with error_log.open("a", encoding="utf-8") as fh:
                fh.write(f"{job['job_id']}\t{type(last_error).__name__}: {last_error}\n")
            print(f"[skip] {job['job_id']} 评估失败：{last_error}")
            continue
        append_jsonl(output_dir / "evaluations.jsonl", {"evaluation_id": stable_hash({"job_id": job["job_id"], "prompt": prompt}), "job_id": job["job_id"], "trajectory_id": job["trajectory_id"], "story_id": job["story_id"], "episode_id": job["episode_id"], "condition": job["condition"], "blind": True, "scores": {field: result[field] for field in SCORE_FIELDS}, "details": result["details"], "justifications": result["justifications"], "rule_echo_spans": result.get("rule_echo_spans", []), "violations": result["violations"], "created_at": utc_now()})
        completed += 1
    print(f"[done] completed={completed} skipped={skipped}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    parser.add_argument("--fake-api", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stories", type=str, help="逗号分隔，如 S02,S03")
    parser.add_argument("--runs", type=str, help="逗号分隔，如 R01,R02")
    args = parser.parse_args()
    story_filter = set(args.stories.split(",")) if args.stories else None
    run_filter = set(args.runs.split(",")) if args.runs else None
    count = run(args.output_dir, fake_api=args.fake_api, execute_api=args.execute_api, limit=args.limit, story_filter=story_filter, run_filter=run_filter)
    print(f"evaluated_jobs={count} fake_api={args.fake_api}")


if __name__ == "__main__":
    main()
