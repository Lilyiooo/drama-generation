#!/usr/bin/env python3
"""多 Agent 评测的稳定性与区分度验证工具。

同一批剧本重复运行多 Agent Demo，或仅统计已有结果，输出四项核心指标：
- WithinStd：同一剧本多轮最终分的平均标准差，越低越稳定；
- BetweenStd：同轮不同剧本最终分的平均标准差，越高表示区分度仍被保留；
- RankSpearman：不同轮次剧本排名的平均 Spearman 相关，越高越稳定；
- DiscriminationRatio：BetweenStd / (WithinStd + epsilon)，越高越好。

用法：
  python measure_multi_agent.py 剧本A.txt 剧本B.txt --runs 3 --output-root output/multi_agent_stability
  python measure_multi_agent.py --output-root output/multi_agent_stability
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCORE_LABELS = ["剧本逻辑总分", "剧本质量最终总分", "剧本创意总分"]
EVALUATE_MULTI_AGENT = Path(__file__).resolve().parent / "evaluate_multi_agent.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="多 Agent 评测稳定性与区分度验证。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scripts", nargs="*", type=Path, help="待测剧本 txt/md 文件；省略时只统计已有结果")
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="每个剧本对应的目录前缀；同名剧本必须显式提供以免结果混组。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "multi_agent_stability",
        help="每轮输出为 <root>/<前缀>_Rxx/。",
    )
    parser.add_argument("--runs", type=int, default=3, help="每个剧本的独立运行轮数")
    parser.add_argument("--model", default="gpt-5.6-sol", help="评测模型")
    parser.add_argument("--reviewers", type=int, default=3, help="每轮独立评审数量")
    parser.add_argument("--agent-workers", type=int, default=1, help="单轮独立评审并发数")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="同时运行的剧本轮次数；大于 1 会叠加 API 并发，谨慎使用。",
    )
    parser.add_argument("--sleep", type=float, default=12.0, help="串行相邻任务的等待秒数")
    parser.add_argument("--restart", action="store_true", help="忽略已有结果并完整重跑")
    parser.add_argument("--no-files", action="store_true", help="仅打印，不写 JSON/Markdown 汇总")
    return parser


def run_one_eval(
    script: Path,
    output_dir: Path,
    model: str,
    reviewers: int,
    agent_workers: int,
    restart: bool,
) -> Path:
    command = [
        sys.executable,
        str(EVALUATE_MULTI_AGENT),
        str(script),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--reviewers",
        str(reviewers),
        "--agent-workers",
        str(agent_workers),
    ]
    if restart:
        command.append("--restart")
    log_path = Path(f"{output_dir}.log")
    with open(log_path, "w", encoding="utf-8") as file:
        result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
        raise RuntimeError(f"评测失败 exit={result.returncode}：\n" + "\n".join(tail))
    return output_dir / "scores.json"


def discover_groups(output_root: Path) -> dict[str, dict[int, Path]]:
    groups: dict[str, dict[int, Path]] = {}
    for path in sorted(output_root.glob("*/scores.json")):
        name = path.parent.name
        marker = name.rsplit("_R", 1)
        if len(marker) == 2 and marker[1].isdigit():
            prefix, run = marker[0], int(marker[1])
        else:
            prefix, run = name, 0
        groups.setdefault(prefix, {})[run] = path
    return groups


def load_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    if not isinstance(scores, dict):
        return {}
    return {
        label: float(value)
        for label, value in scores.items()
        if label in SCORE_LABELS and isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def population_std(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def rank(values: list[float]) -> list[float]:
    """返回升序平均秩，处理并列值。"""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for offset in range(position, end):
            result[indexed[offset][0]] = average_rank
        position = end
    return result


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if not left_scale or not right_scale:
        return None
    return numerator / (left_scale * right_scale)


def spearman(left: list[float], right: list[float]) -> float | None:
    return pearson(rank(left), rank(right))


def summarize(groups: dict[str, dict[int, Path]]) -> dict[str, Any]:
    cached_scores = {path: load_scores(path) for runs in groups.values() for path in runs.values()}
    dimensions: dict[str, dict[str, Any]] = {}
    for label in SCORE_LABELS:
        per_script: dict[str, dict[int, float]] = {}
        for prefix, runs in groups.items():
            values = {
                run: cached_scores[path][label]
                for run, path in runs.items()
                if label in cached_scores[path]
            }
            if values:
                per_script[prefix] = values

        within_by_script = {
            prefix: population_std(list(values.values()))
            for prefix, values in per_script.items()
            if len(values) >= 2
        }
        all_runs = sorted({run for values in per_script.values() for run in values})
        between_by_run = {
            run: population_std([values[run] for values in per_script.values() if run in values])
            for run in all_runs
            if sum(1 for values in per_script.values() if run in values) >= 2
        }

        pairwise_ranks: dict[str, float] = {}
        for index, left_run in enumerate(all_runs):
            for right_run in all_runs[index + 1 :]:
                common = [
                    prefix
                    for prefix, values in per_script.items()
                    if left_run in values and right_run in values
                ]
                correlation = spearman(
                    [per_script[prefix][left_run] for prefix in common],
                    [per_script[prefix][right_run] for prefix in common],
                )
                if correlation is not None:
                    pairwise_ranks[f"R{left_run:02d}-R{right_run:02d}"] = round(correlation, 4)

        within = statistics.mean(within_by_script.values()) if within_by_script else None
        between = statistics.mean(between_by_run.values()) if between_by_run else None
        dimensions[label] = {
            "per_script": {
                prefix: {
                    "values": {f"R{run:02d}": round(score, 2) for run, score in sorted(values.items())},
                    "within_std": round(within_by_script[prefix], 4) if prefix in within_by_script else None,
                }
                for prefix, values in sorted(per_script.items())
            },
            "within_std": round(within, 4) if within is not None else None,
            "between_std": round(between, 4) if between is not None else None,
            "between_std_by_run": {f"R{run:02d}": round(value, 4) for run, value in between_by_run.items()},
            "rank_spearman": pairwise_ranks,
            "rank_spearman_mean": round(statistics.mean(pairwise_ranks.values()), 4) if pairwise_ranks else None,
            "discrimination_ratio": round(between / (within + 1e-6), 4)
            if within is not None and between is not None
            else None,
        }
    return {
        "score_labels": SCORE_LABELS,
        "script_count": len(groups),
        "groups": {prefix: sorted(runs) for prefix, runs in groups.items()},
        "dimensions": dimensions,
        "interpretation": {
            "within_std": "同一剧本重测波动，越低越稳定。",
            "between_std": "同轮不同剧本得分的离散度，过度降低意味着区分度被压缩。",
            "rank_spearman": "不同轮次剧本排名的 Spearman 相关，越接近 1 越稳定。",
            "discrimination_ratio": "BetweenStd / (WithinStd + epsilon)，越高越能兼顾区分度与稳定性。",
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    rows = []
    for label, result in summary["dimensions"].items():
        def fmt(value: object) -> str:
            return "-" if value is None else f"{float(value):.3f}"

        rows.append(
            f"| {label} | {fmt(result['within_std'])} | {fmt(result['between_std'])} | "
            f"{fmt(result['rank_spearman_mean'])} | {fmt(result['discrimination_ratio'])} |"
        )
    return "\n".join(
        [
            "# 多 Agent 稳定性与区分度验证",
            "",
            f"- 剧本数：{summary['script_count']}",
            "",
            "| 维度 | WithinStd ↓ | BetweenStd（不应被压缩） | Rank Spearman ↑ | DiscriminationRatio ↑ |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            "- `WithinStd`：同一剧本多轮结果的平均总体标准差。",
            "- `BetweenStd`：每轮不同剧本得分的总体标准差，再跨轮平均。",
            "- `Rank Spearman`：每对轮次的剧本排名相关系数，再跨轮对平均；至少需要两部共同剧本。",
            "- `DiscriminationRatio = BetweenStd / (WithinStd + 1e-6)`：数值越高，表示稳定性改善而剧本间差异仍被保留。",
            "",
            "> 该工具只报告统计事实；是否优于单 Agent，应使用同一批剧本、相同轮数的单 Agent 对照结果比较。",
        ]
    )


def main() -> int:
    args = build_parser().parse_args()
    if not EVALUATE_MULTI_AGENT.is_file():
        print(f"错误：未找到 {EVALUATE_MULTI_AGENT}", file=sys.stderr)
        return 1
    if args.runs < 1 or args.reviewers < 2 or args.parallel < 1 or args.agent_workers < 1:
        print("错误：runs、reviewers、parallel、agent-workers 必须满足最小值。", file=sys.stderr)
        return 1
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.scripts:
        prefixes = list(args.prefix) if args.prefix else []
        if prefixes and len(prefixes) != len(args.scripts):
            print("错误：--prefix 数量必须与 scripts 一致。", file=sys.stderr)
            return 1
        jobs: list[tuple[Path, Path]] = []
        for index, script in enumerate(args.scripts):
            if not script.is_file():
                print(f"警告：剧本不存在，跳过：{script}", file=sys.stderr)
                continue
            prefix = prefixes[index] if prefixes else script.stem
            for run in range(1, args.runs + 1):
                jobs.append((script.expanduser().resolve(), args.output_root / f"{prefix}_R{run:02d}"))

        def should_run(path: Path) -> bool:
            return args.restart or not (path / "scores.json").exists()

        pending = [(script, output) for script, output in jobs if should_run(output)]
        if args.parallel == 1:
            for index, (script, output) in enumerate(jobs, start=1):
                if not should_run(output):
                    print(f"[{index}/{len(jobs)}] 复用 {output.name}", flush=True)
                    continue
                print(f"[{index}/{len(jobs)}] 评测 {output.name} ...", flush=True)
                try:
                    run_one_eval(script, output, args.model, args.reviewers, args.agent_workers, args.restart)
                except RuntimeError as error:
                    print(f"  失败：{error}", file=sys.stderr, flush=True)
                if args.sleep and index < len(jobs):
                    time.sleep(args.sleep)
        else:
            print(f"并行评测（--parallel={args.parallel}），可能触发 TPM 限流。", flush=True)
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(
                        run_one_eval,
                        script,
                        output,
                        args.model,
                        args.reviewers,
                        args.agent_workers,
                        args.restart,
                    ): output
                    for script, output in pending
                }
                for future in as_completed(futures):
                    output = futures[future]
                    try:
                        future.result()
                        print(f"完成 {output.name}", flush=True)
                    except RuntimeError as error:
                        print(f"失败 {output.name}：{error}", file=sys.stderr, flush=True)

    groups = discover_groups(args.output_root)
    if not groups:
        print(f"未在 {args.output_root} 下找到 scores.json。", file=sys.stderr)
        return 1
    summary = summarize(groups)
    print(render_markdown(summary))
    if not args.no_files:
        json_path = args.output_root / "multi_agent_stability_discrimination.json"
        markdown_path = args.output_root / "multi_agent_stability_discrimination.md"
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        markdown_path.write_text(render_markdown(summary) + "\n", encoding="utf-8")
        print(f"\n汇总已写出：{json_path}")
        print(f"汇总已写出：{markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
