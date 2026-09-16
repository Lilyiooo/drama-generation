from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .io_utils import read_jsonl, utc_now, write_json
from .schema import ROOT


def entropy(values: Iterable[str]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def gini(values: Iterable[int]) -> float:
    ordered = sorted(value for value in values if value >= 0)
    if not ordered or sum(ordered) == 0:
        return 0.0
    n = len(ordered)
    return sum((2 * index - n - 1) * value for index, value in enumerate(ordered, start=1)) / (n * sum(ordered))


def effective_count(values: Iterable[str]) -> float:
    return 2 ** entropy(values)


def job_group(job_id: str) -> tuple[str, str, str]:
    """Derive (condition, story_id, episode_id) from a job id like S01__C_x__R01__E01."""
    parts = job_id.split("__")
    return (parts[1], parts[0], parts[3])


def analyze(run_dir: Path) -> dict[str, Any]:
    extractions = read_jsonl(run_dir / "extractions.jsonl")
    retrievals = read_jsonl(run_dir / "retrieval_events.jsonl")
    evaluations = read_jsonl(run_dir / "evaluations.jsonl")
    branches = read_jsonl(run_dir / "branches.jsonl")
    states = read_jsonl(run_dir / "state_updates.jsonl")
    contexts = read_jsonl(run_dir / "contexts.jsonl")
    writebacks = read_jsonl(run_dir / "writebacks.jsonl")
    moves_by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    moves_by_trajectory: dict[str, list[tuple[str, str]]] = defaultdict(list)
    # per-job adopted move set (primary + secondary) and retrieved cluster set
    moves_by_job: dict[str, set[str]] = {}
    retrieved_by_job: dict[str, set[str]] = {}
    for row in extractions:
        result = row["result"]
        move_set = {result["primary_move_id"]}
        move_set.update(result.get("secondary_move_ids") or [])
        moves_by_job[row["job_id"]] = move_set
    for row in contexts:
        retrieved_by_job[row["job_id"]] = set(row.get("retrieved_cluster_ids") or [])
    for row in extractions:
        move = row["result"]["primary_move_id"]
        key = (row["condition"], row["story_id"], row["episode_id"])
        moves_by_group[key].append(move)
        moves_by_trajectory[row["trajectory_id"]].append((row["episode_id"], move))
    retrieval_by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    retrieval_by_run: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    run_by_episode: dict[tuple[str, str, str], str] = {}
    for row in retrievals:
        if not row.get("selected", True):
            continue
        retrieval_by_group[(row["condition"], row["story_id"], row["episode_id"])].append(row["cluster_id"])
        rkey = (row["condition"], row["story_id"], row["run_id"])
        retrieval_by_run[rkey].append(row["cluster_id"])
        run_by_episode[(row["condition"], row["story_id"], row["episode_id"])] = row["run_id"]
    # L3 experience-adoption: per group, the fraction of jobs whose output move was
    # drawn from the retrieved experience clusters, plus retrieval precision.
    adoption_hit_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    adoption_precision_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in extractions:
        key = (row["condition"], row["story_id"], row["episode_id"])
        retrieved = retrieved_by_job.get(row["job_id"], set())
        if not retrieved:
            continue
        primary = row["result"]["primary_move_id"]
        adopted = len(retrieved & moves_by_job.get(row["job_id"], set()))
        adoption_hit_by_group[key].append(1.0 if primary in retrieved else 0.0)
        adoption_precision_by_group[key].append(adopted / len(retrieved))
    # L3 writeback concentration: cluster entropy and mean card quality per group
    writeback_cluster_by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    writeback_quality_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in writebacks:
        key = job_group(row["job_id"])
        card = row.get("card") or {}
        cluster_id = card.get("cluster_id") or card.get("move_id")
        if cluster_id:
            writeback_cluster_by_group[key].append(cluster_id)
        quality = card.get("quality_score")
        if isinstance(quality, (int, float)):
            writeback_quality_by_group[key].append(float(quality))
    scores_by_group: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in evaluations:
        for field, score in row["scores"].items():
            scores_by_group[(row["condition"], row["story_id"], row["episode_id"])][field].append(float(score))
    branches_by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for row in branches:
        branches_by_group[(row["condition"], row["story_id"], row["episode_id"])].extend(branch["move_id"] for branch in row["branches"] if branch.get("move_id"))
    issues_by_group: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in states:
        issues_by_group[(row["condition"], row["story_id"], row["episode_id"])].append(len(row.get("state_consistency_issues", [])))
    repeat_by_group: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for trajectory_id, pairs in moves_by_trajectory.items():
        ordered = sorted(pairs)
        condition, story_id = trajectory_id.split("__", 2)[:2][1], trajectory_id.split("__", 2)[0]
        previous: str | None = None
        for episode_id, move in ordered:
            repeat_by_group[(condition, story_id, episode_id)].append(int(previous == move) if previous else 0)
            previous = move
    keys = sorted(set(moves_by_group) | set(retrieval_by_group) | set(scores_by_group) | set(branches_by_group))
    rows: list[dict[str, Any]] = []
    for condition, story_id, episode_id in keys:
        key = (condition, story_id, episode_id)
        retrieval_counts = Counter(retrieval_by_group[key])
        run = run_by_episode.get(key)
        rbc = retrieval_by_run.get((condition, story_id, run), []) if run else []
        cluster_counts = Counter(rbc)
        run_retrieval_metrics = {
            "retrieval_cluster_entropy_run": round(entropy(rbc), 6) if rbc else None,
            "retrieval_cluster_gini_run": round(gini(cluster_counts.values()), 6) if rbc else None,
            "top_cluster_share_run": round(max(cluster_counts.values()) / len(rbc), 6) if rbc else None,
            "retrieval_hhi_run": round(sum((c / len(rbc)) ** 2 for c in cluster_counts.values()), 6) if rbc else None,
        }
        row: dict[str, Any] = {
            "condition": condition,
            "story_id": story_id,
            "episode_id": episode_id,
            "n_outputs": len(moves_by_group[key]),
            "move_entropy": round(entropy(moves_by_group[key]), 6),
            "effective_move_count": round(effective_count(moves_by_group[key]), 6),
            "cross_episode_repeat_rate": round(mean(repeat_by_group[key]), 6) if repeat_by_group[key] else None,
            "retrieval_cluster_entropy": round(entropy(retrieval_by_group[key]), 6),
            "retrieval_cluster_gini": round(gini(retrieval_counts.values()), 6),
            **run_retrieval_metrics,
            "effective_branch_count": round(effective_count(branches_by_group[key]), 6) if branches_by_group[key] else None,
            "state_issue_rate": round(mean(int(value > 0) for value in issues_by_group[key]), 6) if issues_by_group[key] else None,
            "experience_adoption_rate": round(mean(adoption_hit_by_group[key]), 6) if adoption_hit_by_group[key] else None,
            "retrieval_precision": round(mean(adoption_precision_by_group[key]), 6) if adoption_precision_by_group[key] else None,
            "writeback_cluster_entropy": round(entropy(writeback_cluster_by_group[key]), 6) if writeback_cluster_by_group[key] else None,
            "writeback_card_quality_mean": round(mean(writeback_quality_by_group[key]), 6) if writeback_quality_by_group[key] else None,
        }
        for field, values in scores_by_group[key].items():
            row[f"mean_{field}"] = round(mean(values), 6)
        rows.append(row)
    output_csv = run_dir / "episode_metrics.csv"
    if rows:
        fields = sorted({field for row in rows for field in row})
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "created_at": utc_now(),
        "metric_definitions": {
            "move_entropy": "同一条件、故事、集数跨重复运行的 primary Narrative Move Shannon entropy",
            "cross_episode_repeat_rate": "轨迹内当前集 primary Move 与上一集相同的比例",
            "retrieval_cluster_entropy": "同一条件、故事、集数的检索簇 Shannon entropy",
            "retrieval_cluster_gini": "检索簇曝光计数 Gini；越高越集中",
            "effective_branch_count": "候选分支 Move 分布的 exp2(entropy)",
            "state_issue_rate": "存在至少一个状态一致性问题的输出比例",
            "experience_adoption_rate": "L3 经验采用率：检索到的经验簇中，primary Move 实际落在其中的输出占比（精度口径，按 job 二值取平均）",
            "retrieval_precision": "L3 检索精度：每 job 被采纳的 Move（primary∪secondary 与检索簇交集）占检索簇比例的平均值",
            "writeback_cluster_entropy": "L3 写回集中度：同组写回卡片 cluster_id 的 Shannon entropy，越低越集中",
            "writeback_card_quality_mean": "L3 写回卡质量分：同组写回卡片 quality_score 均值",
        },
        "episode_metrics": rows,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    args = parser.parse_args()
    summary = analyze(args.run_dir)
    print(f"metric_rows={len(summary['episode_metrics'])}")


if __name__ == "__main__":
    main()
