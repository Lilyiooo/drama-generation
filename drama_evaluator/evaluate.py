from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from drama_evaluator import EvaluationConfig, EvaluationPipeline


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            os.environ.setdefault(key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="输入一个剧本，使用 GPT-5.6 生成多维度评估报告。"
    )
    parser.add_argument("script", type=Path, help="待评估剧本 txt/md 文件路径")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="结果目录；默认 output/<剧本文件名>",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("DRAMA_EVAL_MODEL", "gpt-5.6-sol"),
        help="模型名，默认 gpt-5.6-sol；请求方式复用 drama_operator_new",
    )
    parser.add_argument("--workers", type=int, default=3, help="最大并发请求数")
    parser.add_argument(
        "--max-output-tokens", type=int, default=16000, help="单次评估最大输出 token"
    )
    parser.add_argument("--timeout", type=float, default=600.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数")
    parser.add_argument(
        "--context-mode",
        choices=("auto", "direct", "evidence"),
        default="auto",
        help="direct=全文直送；evidence=先分段提取证据；auto=按长度自动选择",
    )
    parser.add_argument(
        "--direct-char-limit",
        type=int,
        default=300000,
        help="auto 模式下全文直送的最大字符数",
    )
    parser.add_argument(
        "--chunk-chars",
        type=int,
        default=100000,
        help="证据提取模式下每个连续剧本片段的目标字符数",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="清除该结果目录内已生成的评估产物并重新运行",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查输入、提示词和预计调用量，不请求模型",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "output" / args.script.stem
    config = EvaluationConfig(
        script_path=args.script.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        model=args.model,
        workers=args.workers,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
        context_mode=args.context_mode,
        direct_char_limit=args.direct_char_limit,
        chunk_chars=args.chunk_chars,
        restart=args.restart,
    )
    try:
        pipeline = EvaluationPipeline(config)
        if args.dry_run:
            print(json.dumps(pipeline.workload(), ensure_ascii=False, indent=2))
            return 0
        summary_path = pipeline.run()
        print(f"评估完成：{summary_path}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
