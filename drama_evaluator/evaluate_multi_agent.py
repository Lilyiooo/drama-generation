from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

from drama_evaluator import MultiAgentEvaluationConfig, MultiAgentEvaluationPipeline


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
        description="三大维度各 3 评审 + 各 1 审核 + 各按需 1 仲裁的 S/A/B 台账评估。"
    )
    parser.add_argument("script", type=Path, help="待评估剧本 txt/md 文件路径")
    parser.add_argument(
        "--output-dir", type=Path,
        help="结果目录；默认 output/dimension_multi_agent_<剧本文件名>_<模型名>",
    )
    parser.add_argument(
        "--model", default=os.getenv("DRAMA_EVAL_MODEL", "gpt-5.6-sol"),
        help="评测模型名，默认 gpt-5.6-sol",
    )
    parser.add_argument("--review-workers", type=int, default=3, help="独立评审的最大并发数")
    parser.add_argument("--audit-workers", type=int, default=3, help="3 个维度审核的最大并发数")
    parser.add_argument("--arbitration-workers", type=int, default=3, help="最多 3 个维度仲裁的最大并发数")
    parser.add_argument("--max-output-tokens", type=int, default=16000, help="单次最大输出 token")
    parser.add_argument("--timeout", type=float, default=600.0, help="单次请求超时秒数")
    parser.add_argument("--retries", type=int, default=3, help="API 请求重试次数")
    parser.add_argument("--parse-retries", type=int, default=2, help="JSON 格式重试次数")
    parser.add_argument(
        "--context-mode", choices=("auto", "direct", "evidence"), default="auto",
        help="direct=全文直送；evidence=先分段提取证据；auto=按长度选择",
    )
    parser.add_argument("--direct-char-limit", type=int, default=300000)
    parser.add_argument("--chunk-chars", type=int, default=100000)
    parser.add_argument("--restart", action="store_true", help="清除该目录内产物并重新评估")
    parser.add_argument("--dry-run", action="store_true", help="仅检查并展示预计调用量")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        safe_model = args.model.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_dir = (
            Path(__file__).resolve().parent / "output"
            / f"dimension_multi_agent_{args.script.stem}_{safe_model}"
        )
    config = MultiAgentEvaluationConfig(
        script_path=args.script.expanduser().resolve(),
        output_dir=output_dir.expanduser().resolve(),
        model=args.model,
        review_workers=args.review_workers,
        audit_workers=args.audit_workers,
        arbitration_workers=args.arbitration_workers,
        max_output_tokens=args.max_output_tokens,
        timeout=args.timeout,
        retries=args.retries,
        parse_retries=args.parse_retries,
        context_mode=args.context_mode,
        direct_char_limit=args.direct_char_limit,
        chunk_chars=args.chunk_chars,
        restart=args.restart,
    )
    try:
        pipeline = MultiAgentEvaluationPipeline(config)
        if args.dry_run:
            print(json.dumps(pipeline.workload(), ensure_ascii=False, indent=2))
            return 0
        report = pipeline.run()
        print(f"多 Agent 台账评估完成：{report}")
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
