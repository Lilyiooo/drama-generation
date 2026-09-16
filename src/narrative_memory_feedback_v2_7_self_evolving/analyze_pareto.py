"""双记忆实验帕累托分析：合并多 run 目录，按条件聚合「重复代价 vs 一致性收益 vs 质量/新颖」。

设计目标（对应研究主线）：在长篇创意写作中，相似性检索是否形成「检索诱导的叙事同质化」，
使局部相关性（Rel）与全局创造性（Novelty）冲突。本脚本把各条件放到同一张表，做帕累托比较：

  重复代价轴   : SRI、跨集 4-gram 复用、后期-前期 复用差（趋势）、move 跨集重复率、
                 move 熵、检索簇熵
  一致性收益轴 : state_consistency、causal_continuity、anchor_satisfaction、state_issue_rate(越低越好)
  质量/新颖轴  : dialogue_quality、scene_structure、mechanism_novelty、experience_adoption_rate

用法:
  python -m narrative_memory_feedback_v2_2.analyze_pareto [--runs-dir <dir>] [--conditions A,B,D,...] [--output <csv>]

扫描 runs-dir 下所有含 generations.jsonl 的目录，按 condition 跨目录合并
（A/B/D 在历史分目录；新臂 D_diverse/D_random/D_no_card 在同一新目录）。
直接、容错地读取各 jsonl，避免旧目录 retrieval_events 缺 run_id 导致 analyze 整体崩溃。
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .analyze import entropy, gini
from .io_utils import read_jsonl
from .schema import POOL_TAG_BY_CONDITION, ROOT
from .surface_repetition import analyze as analyze_surface

RUNS = ROOT / "runs"

EXPECTED_CONDITIONS = [
    "A_state_only", "B_high_diverse_frozen", "D_high_diverse_append",
    "D_diverse", "D_random", "D_no_card",
]

CONSISTENCY_FIELDS = ["state_consistency", "causal_continuity", "anchor_satisfaction"]
QUALITY_FIELDS = ["dialogue_quality", "scene_structure", "mechanism_novelty", "plot_coherence", "constraint_satisfaction"]


def _mean(values: list[float]) -> float | None:
    vals = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(vals), 4) if vals else None


def collect(runs_dir: Path, condition_filter: list[str] | None, *, dedup: bool = True) -> dict[str, dict[str, list[float]]]:
    """扫描 runs-dir 下所有含 generations.jsonl 的目录，按 condition 跨目录合并。

    当 dedup=True 时，以 job_id 去重 evaluation/state/extraction/retrieval 记录，
    以 trajectory_id 去重 surface_repetition 轨迹，避免跨目录同名 job 被重复计入。
    """
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen_trajs: set[str] = set()
    seen_jobs: set[str] = set()

    def want(cond: str | None) -> bool:
        return bool(cond) and (condition_filter is None or cond in condition_filter)

    # 按 generations 行数降序排列：完整目录先处理，确保 dedup 时更完整的数据被保留
    dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or not (d / "generations.jsonl").exists():
            continue
        if d.name in {"nmf_dual_memory_smoke"}:
            continue
        gen_count = sum(1 for _ in open(d / "generations.jsonl")) if (d / "generations.jsonl").exists() else 0
        dirs.append((d, gen_count))
    dirs.sort(key=lambda x: -x[1])
    for run_dir, _gen_count in dirs:

        # 表层重复（surface_repetition 不依赖 run_id，稳健）
        try:
            s = analyze_surface(run_dir)
            for traj in s.get("trajectories", []):
                cond = traj["trajectory_id"].split("__")[1]
                if not want(cond):
                    continue
                if dedup and traj["trajectory_id"] in seen_trajs:
                    continue
                seen_trajs.add(traj["trajectory_id"])
                buckets[cond]["sri"].append(traj["surface_repetition_index"])
                buckets[cond]["sri_legacy"].append(traj.get("surface_repetition_index_legacy", traj["surface_repetition_index"]))
                buckets[cond]["global_pairwise_jaccard"].append(traj.get("global_pairwise_4gram_jaccard", traj.get("opening_template_score", 0)))
                buckets[cond]["cross_ep_4gram_rep"].append(traj["mean_cross_ep_4gram_rep"])
                buckets[cond]["opening_template_score"].append(traj["opening_template_score"])
                buckets[cond]["formulaic_density_pct"].append(traj.get("formulaic_density_pct", 0))
                buckets[cond]["SRI_v3"].append(traj.get("surface_repetition_index_v3", 0))
                buckets[cond]["boundary_mean_score"].append(traj.get("boundary_mean_score", 0))
                buckets[cond]["boundary_high_count"].append(traj.get("boundary_high_count", 0))
                trend = traj.get("trend_early_vs_late") or {}
                late = trend.get("late_cross_ep_4gram_rep")
                early = trend.get("early_cross_ep_4gram_rep")
                if isinstance(late, (int, float)) and isinstance(early, (int, float)):
                    buckets[cond]["late_minus_early_rep"].append(late - early)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] surface_repetition 失败 {run_dir.name}: {exc}")

        # 质量 / 一致性（evaluations）
        ev_path = run_dir / "evaluations.jsonl"
        if ev_path.exists():
            for row in read_jsonl(ev_path):
                cond = row.get("condition")
                jid = row.get("job_id")
                if not want(cond):
                    continue
                if dedup and jid and jid in seen_jobs:
                    continue
                if jid:
                    seen_jobs.add(jid)
                for f, v in (row.get("scores") or {}).items():
                    if isinstance(v, (int, float)):
                        buckets[cond][f"mean_{f}"].append(float(v))

        # 状态一致性问题率（states）
        st_path = run_dir / "state_updates.jsonl"
        state_jobs: set[str] = set()
        if st_path.exists():
            for row in read_jsonl(st_path):
                cond = row.get("condition")
                jid = row.get("job_id")
                if not want(cond):
                    continue
                if dedup and jid and jid in state_jobs:
                    continue
                if jid:
                    state_jobs.add(jid)
                issues = row.get("state_consistency_issues") or []
                buckets[cond]["state_issue_rate"].append(1.0 if len(issues) > 0 else 0.0)

        # 叙事机制（extractions）：move 熵 + 跨集 move 重复
        ex_path = run_dir / "extractions.jsonl"
        by_traj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        by_group: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        retrieved_by_job: dict[str, set[str]] = {}
        moves_by_job: dict[str, set[str]] = {}
        moves_by_job_cond: dict[str, str] = {}
        extra_jobs: set[str] = set()
        if ex_path.exists():
            for row in read_jsonl(ex_path):
                cond = row.get("condition")
                jid = row.get("job_id")
                if not want(cond):
                    continue
                if dedup and jid and jid in extra_jobs:
                    continue
                if jid:
                    extra_jobs.add(jid)
                move = row["result"]["primary_move_id"]
                by_group[(cond, row.get("story_id"), row.get("episode_id"))].append(move)
                by_traj[row.get("trajectory_id")].append((row.get("episode_id"), move))
                moves_by_job[jid] = {move}
            for (_c, _s, _e), moves in by_group.items():
                buckets[_c]["move_entropy"].append(entropy(moves))
            for traj_id, pairs in by_traj.items():
                cond = traj_id.split("__")[1]
                if not want(cond):
                    continue
                ordered = sorted(pairs)
                prev: str | None = None
                reps: list[float] = []
                for _ep, move in ordered:
                    if prev is not None:
                        reps.append(1.0 if prev == move else 0.0)
                    prev = move
                if reps:
                    buckets[cond]["cross_episode_repeat_rate"].append(statistics.mean(reps))

        # 检索层（retrieval_events，容错缺 run_id）
        rt_path = run_dir / "retrieval_events.jsonl"
        rt_jobs: set[str] = set()
        if rt_path.exists():
            by_run: dict[tuple[str, str, str], list[str]] = defaultdict(list)
            adoption_hit: dict[str, list[float]] = defaultdict(list)
            for row in read_jsonl(rt_path):
                cond = row.get("condition")
                jid = row.get("job_id")
                if not want(cond) or not row.get("selected", True):
                    continue
                if dedup and jid and jid in rt_jobs:
                    continue
                if jid:
                    rt_jobs.add(jid)
                rid = row.get("run_id")
                story = row.get("story_id")
                by_run[(cond, story, rid)].append(row.get("cluster_id"))
                retrieved_by_job.setdefault(jid, set()).add(row.get("cluster_id"))
                # 先记录 adoption_hit 的 condition（每条 retrieval_event 的 cond），
                # 后面按 jid 去重后统一计算
                if jid not in moves_by_job_cond:
                    moves_by_job_cond[jid] = cond
            for (_c, _s, _r), clusters in by_run.items():
                buckets[_c]["retrieval_cluster_entropy"].append(entropy(clusters))
    # ── 跨全局 adoption 汇总（在所有目录处理完后统一计算） ──
    adoption_hit: dict[str, list[float]] = defaultdict(list)
    adoption_done: set[str] = set()
    for jid, move_set in moves_by_job.items():
        if jid in adoption_done:
            continue
        adoption_done.add(jid)
        retr = retrieved_by_job.get(jid, set())
        if not retr:
            continue
        cond_match = moves_by_job_cond.get(jid)
        if not want(cond_match):
            continue
        primary = next(iter(move_set), None)
        adoption_hit[cond_match].append(1.0 if (primary and primary in retr) else 0.0)
    for cond, lst in adoption_hit.items():
        buckets[cond]["experience_adoption_rate"].extend(lst)
    return buckets


def build_table(buckets: dict[str, dict[str, list[float]]]) -> list[dict[str, Any]]:
    rows = []
    ordered = [c for c in EXPECTED_CONDITIONS if c in buckets] + [c for c in buckets if c not in EXPECTED_CONDITIONS]
    for cond in ordered:
        b = buckets[cond]
        rows.append({
            "condition": cond,
            "pool": POOL_TAG_BY_CONDITION.get(cond) or "none",
            "SRI": _mean(b["sri"]),
            "SRI_v3": _mean(b.get("SRI_v3", [])),
            "formulaic_density": _mean(b.get("formulaic_density_pct", [])),
            "boundary_mean_score": _mean(b.get("boundary_mean_score", [])),
            "boundary_high_count": _mean(b.get("boundary_high_count", [])),
            "cross_ep_4gram_rep": _mean(b["cross_ep_4gram_rep"]),
            "late-early_rep": _mean(b["late_minus_early_rep"]),
            "move_entropy": _mean(b["move_entropy"]),
            "cross_ep_move_repeat": _mean(b["cross_episode_repeat_rate"]),
            "retrieval_cluster_entropy": _mean(b["retrieval_cluster_entropy"]),
            "state_consistency": _mean(b["mean_state_consistency"]),
            "causal_continuity": _mean(b["mean_causal_continuity"]),
            "anchor_satisfaction": _mean(b["mean_anchor_satisfaction"]),
            "state_issue_rate": _mean(b["state_issue_rate"]),
            "dialogue_quality": _mean(b["mean_dialogue_quality"]),
            "scene_structure": _mean(b["mean_scene_structure"]),
            "mechanism_novelty": _mean(b["mean_mechanism_novelty"]),
            "experience_adoption": _mean(b["experience_adoption_rate"]),
        })
    return rows


def print_markdown(rows: list[dict[str, Any]]) -> None:
    cols = list(rows[0].keys()) if rows else []
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=RUNS)
    parser.add_argument("--conditions", type=str, default=None, help="逗号分隔的条件过滤")
    parser.add_argument("--output", type=Path, default=None, help="可选 CSV 输出路径")
    parser.add_argument("--no-dedup", action="store_true", help="禁用跨目录去重（保持旧行为）")
    args = parser.parse_args()
    condition_filter = [c.strip() for c in args.conditions.split(",")] if args.conditions else None
    buckets = collect(args.runs_dir, condition_filter, dedup=not args.no_dedup)
    rows = build_table(buckets)
    print(f"# 双记忆实验帕累托对比（扫描 {args.runs_dir}）\n")
    print("读法：重复代价列（SRI / SRI_v3 / formulaic_density / cross_ep_4gram_rep / late-early_rep / move_entropy / "
          "cross_ep_move_repeat / retrieval_cluster_entropy）越低越好；"
          "一致性/质量列越高越好；state_issue_rate 越低越好。\n")
    print_markdown(rows)
    if args.output:
        with args.output.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[written] {args.output}")


if __name__ == "__main__":
    main()
