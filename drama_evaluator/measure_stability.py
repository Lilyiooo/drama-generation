#!/usr/bin/env python3
"""单模型评测波动测量脚本（可复用）。

对同一批剧本，用相同 prompt + 模型独立评测多轮，统计各维度分数波动
（均值 / 标准差 / 极差 / 变异系数 CV%），用于判断单模型评分的稳定性。

用法：
  # 跑 5 轮并统计
  python measure_stability.py 剧本A.txt 剧本B.txt --runs 5 --output-root output/stability

  # 只统计已有结果（不调用模型，复用 output-root 下已有的 *_Rxx/scores.json）
  python measure_stability.py --output-root output

依赖：evaluate.py（同目录）。脚本以子进程方式调用它完成单轮评测，
并复用其断点续跑能力（已完成的专项报告会跳过，只补缺失步骤）。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCORE_LABELS = [
    "剧本创意总分",
    "剧本逻辑总分",
    "人物质量总分",
    "卡点质量总分",
    "情节规划质量总分",
    "分集剧本质量总分",
    "设定质量总分",
    "剧本质量最终总分",
]

EVALUATE_PY = Path(__file__).resolve().parent / "evaluate.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="单模型评测波动测量：同一剧本多轮独立评测并统计维度波动。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "scripts",
        nargs="*",
        type=Path,
        help="剧本 txt/md 文件路径（可多个）。省略则只统计 --output-root 下已有结果。",
    )
    parser.add_argument(
        "--prefix",
        nargs="*",
        default=None,
        help="每个剧本的自定义目录前缀，与 scripts 一一对应；省略则用文件名 stem。"
        "当多个剧本文件名相同（如都叫 scripts.txt）时必须用它区分。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "stability",
        help="评测结果根目录。每个剧本每轮生成 <root>/<剧本名>_Rxx/。",
    )
    parser.add_argument("--runs", type=int, default=5, help="每个剧本独立评测轮数")
    parser.add_argument("--model", default="gpt-5.6-sol", help="评测模型名")
    parser.add_argument("--workers", type=int, default=3, help="单轮评测内部并发请求数")
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="同时评测的剧本数。>1 会成倍增加并发，可能触发 Venus TPM 限流（经验：5 进程即 429）。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=12.0,
        help="串行模式下相邻轮次之间的间隔秒数，用于避让 TPM 限流。",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="清除已生成的评估产物并重新评测（忽略断点续跑）。",
    )
    parser.add_argument(
        "--no-files",
        action="store_true",
        help="不写出 CSV / JSON 汇总文件，只在控制台打印。",
    )
    return parser


def run_one_eval(
    script: Path,
    output_dir: Path,
    model: str,
    workers: int,
    restart: bool,
) -> Path:
    """调用 evaluate.py 完成单轮评测，返回 scores.json 路径。"""
    cmd = [
        sys.executable,
        str(EVALUATE_PY),
        str(script),
        "--output-dir",
        str(output_dir),
        "--model",
        model,
        "--workers",
        str(workers),
    ]
    if restart:
        cmd.append("--restart")
    log_path = Path(str(output_dir) + ".log")
    with open(log_path, "w", encoding="utf-8") as fp:
        result = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-8:]
        raise RuntimeError(f"评测失败 exit={result.returncode}：\n" + "\n".join(tail))
    return output_dir / "scores.json"


def discover_groups(output_root: Path) -> dict[str, dict[int, Path]]:
    """扫描 output_root 下所有 scores.json，按 `<前缀>_R<数字>` 分组。

    返回 {前缀: {轮次: scores.json 路径}}。
    """
    groups: dict[str, dict[int, Path]] = {}
    for scores_path in sorted(output_root.glob("*/scores.json")):
        name = scores_path.parent.name
        match = re.match(r"^(.*)_R(\d+)$", name)
        if match:
            prefix, run = match.group(1), int(match.group(2))
        else:
            prefix, run = name, 0
        groups.setdefault(prefix, {})[run] = scores_path
    return groups


def _load_scores(path: Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp).get("scores", {})


def compute_stats(values: list[float]) -> dict[str, float]:
    """给定一个维度的多轮分数，返回统计量。使用总体标准差，与既有结论一致。"""
    n = len(values)
    mean = statistics.mean(values)
    return {
        "n": n,
        "mean": mean,
        "std": statistics.pstdev(values) if n > 1 else 0.0,
        "range": max(values) - min(values),
        "cv_pct": (statistics.pstdev(values) / mean * 100) if n > 1 and mean else 0.0,
    }


def summarize(groups: dict[str, dict[int, Path]]) -> list[dict]:
    """按剧本 × 维度汇总波动统计。"""
    rows: list[dict] = []
    for prefix in sorted(groups):
        runs = groups[prefix]
        ordered_runs = sorted(runs)
        for label in SCORE_LABELS:
            values: list[float] = []
            missing: list[int] = []
            for run in ordered_runs:
                scores = _load_scores(runs[run])
                if label in scores:
                    values.append(float(scores[label]))
                else:
                    missing.append(run)
            stats = compute_stats(values) if values else {
                "n": 0, "mean": 0.0, "std": 0.0, "range": 0.0, "cv_pct": 0.0,
            }
            rows.append({
                "script": prefix,
                "dimension": label,
                "runs": ordered_runs,
                "values": {run: float(_load_scores(runs[run])[label])
                           for run in ordered_runs if label in _load_scores(runs[run])},
                "missing_runs": missing,
                **stats,
            })
    return rows


def _fmt_run_cell(prefix: str, run: int, value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}"


def print_table(prefix: str, runs: list[int], rows: list[dict]) -> None:
    header = f"{'维度':<14}" + "".join(f"{f'R{r:02d}':>8}" for r in runs) + f"{'极差':>7}{'标准差':>7}{'均值':>7}{'CV%':>7}"
    print(f"\n=== {prefix}（{len(runs)} 轮） ===")
    print(header)
    for row in rows:
        if row["script"] != prefix:
            continue
        cells = "".join(
            f"{_fmt_run_cell(prefix, r, row['values'].get(r)):>8}" for r in runs
        )
        print(
            f"{row['dimension']:<14}{cells}"
            f"{row['range']:>7.1f}{row['std']:>7.2f}{row['mean']:>7.1f}{row['cv_pct']:>7.1f}"
        )
        if row["missing_runs"]:
            print(f"    （缺失轮次：{row['missing_runs']}）")


def write_csv(rows: list[dict], runs_all: list[int], path: Path) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.writer(fp)
        header = ["剧本", "维度"] + [f"R{r:02d}" for r in runs_all] + ["均值", "标准差", "极差", "CV%"]
        writer.writerow(header)
        for row in rows:
            line = [row["script"], row["dimension"]]
            line += [row["values"].get(r, "") for r in runs_all]
            line += [round(row["mean"], 2), round(row["std"], 2),
                     round(row["range"], 2), round(row["cv_pct"], 2)]
            writer.writerow(line)


def write_json(rows: list[dict], path: Path) -> None:
    payload = {
        "score_labels": SCORE_LABELS,
        "rows": [
            {
                "script": row["script"],
                "dimension": row["dimension"],
                "runs": row["runs"],
                "values": row["values"],
                "missing_runs": row["missing_runs"],
                "mean": row["mean"],
                "std": row["std"],
                "range": row["range"],
                "cv_pct": row["cv_pct"],
            }
            for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def main() -> int:
    args = build_parser().parse_args()
    if not EVALUATE_PY.is_file():
        print(f"错误：未找到 {EVALUATE_PY}", file=sys.stderr)
        return 1
    args.output_root.mkdir(parents=True, exist_ok=True)

    # 1) 跑评测（如指定了剧本）
    if args.scripts:
        prefixes = list(args.prefix) if args.prefix else []
        if prefixes and len(prefixes) != len(args.scripts):
            print(
                f"警告：--prefix 数量({len(prefixes)})与剧本数({len(args.scripts)})不一致，"
                "将全部回退为文件名 stem",
                file=sys.stderr,
            )
            prefixes = []
        jobs: list[tuple[Path, Path]] = []
        for i, script in enumerate(args.scripts):
            if not script.is_file():
                print(f"警告：剧本不存在，跳过：{script}", file=sys.stderr)
                continue
            stem = prefixes[i] if prefixes else script.stem
            for run in range(1, args.runs + 1):
                out_dir = args.output_root / f"{stem}_R{run:02d}"
                jobs.append((script.expanduser().resolve(), out_dir))

        def _should_run(out_dir: Path) -> bool:
            return args.restart or not (out_dir / "scores.json").exists()

        pending = [(s, o) for s, o in jobs if _should_run(o)]
        if args.parallel <= 1:
            for idx, (script, out_dir) in enumerate(jobs, 1):
                if not _should_run(out_dir):
                    print(f"[{idx}/{len(jobs)}] 复用 {out_dir.name}", flush=True)
                    continue
                print(f"[{idx}/{len(jobs)}] 评测 {out_dir.name} ...", flush=True)
                try:
                    run_one_eval(script, out_dir, args.model, args.workers, args.restart)
                except RuntimeError as exc:
                    print(f"  失败：{exc}", file=sys.stderr, flush=True)
                if args.sleep and idx < len(jobs):
                    time.sleep(args.sleep)
        else:
            print(f"并行评测（--parallel={args.parallel}），注意 TPM 限流风险", flush=True)
            with ThreadPoolExecutor(max_workers=args.parallel) as executor:
                futures = {
                    executor.submit(run_one_eval, s, o, args.model, args.workers, args.restart): o
                    for s, o in pending
                }
                for future in as_completed(futures):
                    out_dir = futures[future]
                    try:
                        future.result()
                        print(f"完成 {out_dir.name}", flush=True)
                    except RuntimeError as exc:
                        print(f"失败 {out_dir.name}：{exc}", file=sys.stderr, flush=True)

    # 2) 统计
    groups = discover_groups(args.output_root)
    if not groups:
        print(f"未在 {args.output_root} 下找到任何 scores.json。", file=sys.stderr)
        return 1
    rows = summarize(groups)

    for prefix in sorted(groups):
        runs = sorted(groups[prefix])
        print_table(prefix, runs, rows)

    all_runs = sorted({r for g in groups.values() for r in g})
    if not args.no_files:
        csv_path = args.output_root / "stability_summary.csv"
        json_path = args.output_root / "stability_summary.json"
        write_csv(rows, all_runs, csv_path)
        write_json(rows, json_path)
        print(f"\n汇总已写出：{csv_path}")
        print(f"汇总已写出：{json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
