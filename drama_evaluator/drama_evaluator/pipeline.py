from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .llm import OpenAIChatClient


@dataclass(frozen=True, slots=True)
class EvaluationTask:
    key: str
    title: str
    prompt_path: Path
    output_name: str


@dataclass(slots=True)
class EvaluationConfig:
    script_path: Path
    output_dir: Path
    model: str = "gpt-5.6-sol"
    workers: int = 3
    max_output_tokens: int = 16000
    timeout: float = 600.0
    retries: int = 3
    context_mode: str = "auto"
    direct_char_limit: int = 300000
    chunk_chars: int = 100000
    restart: bool = False

    def validate(self) -> None:
        if not self.script_path.is_file():
            raise ValueError(f"剧本文件不存在：{self.script_path}")
        if not self.model.strip():
            raise ValueError("模型名称不能为空")
        if self.workers < 1:
            raise ValueError("workers 必须大于等于 1")
        if self.max_output_tokens < 2000:
            raise ValueError("max_output_tokens 不应低于 2000")
        if self.context_mode not in {"auto", "direct", "evidence"}:
            raise ValueError("context_mode 只能是 auto、direct 或 evidence")
        if self.direct_char_limit < 10000 or self.chunk_chars < 10000:
            raise ValueError("文本长度阈值不应低于 10000 字符")


SCORE_LABELS = (
    "剧本创意总分",
    "剧本逻辑总分",
    "人物质量总分",
    "卡点质量总分",
    "情节规划质量总分",
    "分集剧本质量总分",
    "设定质量总分",
    "剧本质量最终总分",
)


class EvaluationPipeline:
    def __init__(self, config: EvaluationConfig) -> None:
        config.validate()
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.prompt_root = self.root / "prompts"
        self.total_prompt = self.prompt_root / "剧本质量" / "剧本质量总分计分.md"
        self.tasks = self._build_tasks()
        self.client: OpenAIChatClient | None = None

    def _build_tasks(self) -> list[EvaluationTask]:
        specs = (
            ("creativity", "剧本创意", self.prompt_root / "剧本创意.md", "01_剧本创意评估.md"),
            ("logic", "剧本逻辑", self.prompt_root / "剧本逻辑.md", "02_剧本逻辑评估.md"),
            ("character", "人物质量", self.prompt_root / "剧本质量" / "人物质量.md", "03_人物质量评估.md"),
            ("hooks", "卡点质量", self.prompt_root / "剧本质量" / "卡点质量.md", "04_卡点质量评估.md"),
            ("plot", "情节规划质量", self.prompt_root / "剧本质量" / "情节规划质量.md", "05_情节规划质量评估.md"),
            ("episode", "分集剧本质量", self.prompt_root / "剧本质量" / "分集剧本质量.md", "06_分集剧本质量评估.md"),
            ("setting", "设定质量", self.prompt_root / "剧本质量" / "设定质量.md", "07_设定质量评估.md"),
        )
        tasks = [EvaluationTask(*spec) for spec in specs]
        missing = [str(task.prompt_path) for task in tasks if not task.prompt_path.is_file()]
        if not self.total_prompt.is_file():
            missing.append(str(self.total_prompt))
        if missing:
            raise ValueError(f"缺少评估提示词：{', '.join(missing)}")
        return tasks

    @staticmethod
    def _read_text(path: Path) -> str:
        return path.read_text(encoding="utf-8-sig").strip()

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(content)
                if not content.endswith("\n"):
                    file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise

    def _manifest_data(self, script: str) -> dict[str, object]:
        prompt_hashes = {
            task.key: self._sha256(self._read_text(task.prompt_path)) for task in self.tasks
        }
        prompt_hashes["quality_total"] = self._sha256(self._read_text(self.total_prompt))
        return {
            "script_path": str(self.config.script_path.resolve()),
            "script_sha256": self._sha256(script),
            "model": self.config.model,
            "llm_backend": "drama_operator_new/service/shortdrama_by_fiction/tools/llm_services.py",
            "context_mode": self.config.context_mode,
            "direct_char_limit": self.config.direct_char_limit,
            "chunk_chars": self.config.chunk_chars,
            "prompt_sha256": prompt_hashes,
        }

    def _prepare_output(self, expected_manifest: dict[str, object]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = self.config.output_dir / "manifest.json"
        if self.config.restart:
            for task in self.tasks:
                (self.config.output_dir / task.output_name).unlink(missing_ok=True)
            for name in (
                "08_剧本质量总分计分.md",
                "00_评估总览.md",
                "scores.json",
                "全文评估证据索引.md",
            ):
                (self.config.output_dir / name).unlink(missing_ok=True)
            evidence_dir = self.config.output_dir / "evidence"
            if evidence_dir.exists():
                for path in evidence_dir.glob("chunk_*.md"):
                    path.unlink()
        elif manifest_path.exists():
            old = json.loads(self._read_text(manifest_path))
            comparable = {key: old.get(key) for key in expected_manifest}
            if comparable != expected_manifest:
                raise RuntimeError(
                    "输出目录中的输入、模型、上下文配置或提示词已变化；"
                    "请指定新的 --output-dir，或使用 --restart 重新评估"
                )
        self._atomic_write(
            manifest_path,
            json.dumps(expected_manifest, ensure_ascii=False, indent=2),
        )

    @staticmethod
    def _split_script(script: str, chunk_chars: int) -> list[str]:
        episode_starts = list(
            re.finditer(r"(?m)^第\s*[0-9一二三四五六七八九十百零〇两]+\s*集\s*$", script)
        )
        if not episode_starts:
            return [script[index : index + chunk_chars] for index in range(0, len(script), chunk_chars)]

        prefix = script[: episode_starts[0].start()].strip()
        episodes: list[str] = []
        for index, match in enumerate(episode_starts):
            end = episode_starts[index + 1].start() if index + 1 < len(episode_starts) else len(script)
            episodes.append(script[match.start() : end].strip())
        if prefix:
            episodes[0] = f"{prefix}\n\n{episodes[0]}"

        chunks: list[str] = []
        current: list[str] = []
        size = 0
        for episode in episodes:
            if current and size + len(episode) > chunk_chars:
                chunks.append("\n\n".join(current))
                current, size = [], 0
            current.append(episode)
            size += len(episode)
        if current:
            chunks.append("\n\n".join(current))
        return chunks

    def _extract_evidence_chunk(self, index: int, total: int, text: str) -> str:
        assert self.client is not None
        system = """你是剧本评估证据整理员。请只整理输入剧本中的客观证据，不做最终评分，不与原著、小说或外部指令对照。你的输出将提供给后续多个专项评估模型。

必须覆盖：
1. 每集核心事件、阶段目标、冲突、转折、集末钩子和情绪变化；
2. 主角、配角、反派的人设表现、动机、选择、弧光、关系变化与角色功能；
3. 世界观、能力、组织、时空、道具、资源、秘密、角色认知和伏笔状态；
4. 台词风格、短句/说明性台词/模型化表达、可传播台词；
5. 场景动作可拍性、心理或作者叙述、视觉符号、名场面和奇观；
6. 创意亮点、模板化迹象、桥段重复、节奏停滞、主副线变化；
7. 所有可能的前后矛盾、因果缺口、状态错位、规则冲突和无效重复。

每条重要判断必须保留集数、场次、人物和足以核验的短原文引文。不得虚构缺失信息。使用紧凑 Markdown，避免重复转述。"""
        user = (
            f"这是全集剧本的第 {index}/{total} 个连续片段。请生成该片段的综合评估证据索引。\n\n"
            "<SCRIPT_CHUNK>\n"
            f"{text}\n"
            "</SCRIPT_CHUNK>"
        )
        return self.client.text(system, user, label=f"证据提取 {index}/{total}")

    def _build_evidence(self, script: str) -> str:
        chunks = self._split_script(script, self.config.chunk_chars)
        evidence_dir = self.config.output_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        results: dict[int, str] = {}
        pending: list[tuple[int, str]] = []
        for index, chunk in enumerate(chunks, start=1):
            path = evidence_dir / f"chunk_{index:03d}.md"
            if path.exists() and not self.config.restart:
                results[index] = self._read_text(path)
            else:
                pending.append((index, chunk))

        if pending:
            with ThreadPoolExecutor(max_workers=min(self.config.workers, len(pending))) as executor:
                futures = {
                    executor.submit(self._extract_evidence_chunk, index, len(chunks), chunk): index
                    for index, chunk in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    content = future.result()
                    results[index] = content
                    self._atomic_write(evidence_dir / f"chunk_{index:03d}.md", content)

        combined = [
            "# 全文评估证据索引",
            "",
            f"> 原始剧本：`{self.config.script_path}`",
            f"> 原始字符数：{len(script)}；连续片段数：{len(chunks)}。",
            "> 本索引由原剧本分段提取，后续专项评估必须以其中的集数、场次和原文证据为依据。",
        ]
        for index in range(1, len(chunks) + 1):
            combined.extend(("", f"## 连续片段 {index}/{len(chunks)}", "", results[index]))
        evidence = "\n".join(combined)
        self._atomic_write(self.config.output_dir / "全文评估证据索引.md", evidence)
        return evidence

    def _evaluation_source(self, script: str) -> tuple[str, str]:
        use_evidence = self.config.context_mode == "evidence" or (
            self.config.context_mode == "auto" and len(script) > self.config.direct_char_limit
        )
        if use_evidence:
            return "全文分段证据索引", self._build_evidence(script)
        return "全集剧本原文", script

    def _evaluate_task(self, task: EvaluationTask, source_name: str, source: str) -> str:
        assert self.client is not None
        system = self._read_text(task.prompt_path)
        user = (
            f"请评估以下{source_name}。这是原创/通用剧本内部评估，不进行小说、原著、"
            "改编规划或生成指令对照。严格依照系统提示词输出完整 Markdown 报告。\n\n"
            f"<{source_name}>\n{source}\n</{source_name}>"
        )
        return self.client.text(system, user, label=task.title)

    def _run_tasks(self, source_name: str, source: str) -> dict[str, str]:
        reports: dict[str, str] = {}
        pending: list[EvaluationTask] = []
        for task in self.tasks:
            path = self.config.output_dir / task.output_name
            if path.exists() and not self.config.restart:
                print(f"[复用] {task.title}: {path}", flush=True)
                reports[task.key] = self._read_text(path)
            else:
                pending.append(task)

        if pending:
            with ThreadPoolExecutor(max_workers=min(self.config.workers, len(pending))) as executor:
                futures = {
                    executor.submit(self._evaluate_task, task, source_name, source): task
                    for task in pending
                }
                for future in as_completed(futures):
                    task = futures[future]
                    report = future.result()
                    reports[task.key] = report
                    self._atomic_write(self.config.output_dir / task.output_name, report)
        return reports

    def _quality_total(self, reports: dict[str, str]) -> str:
        output = self.config.output_dir / "08_剧本质量总分计分.md"
        if output.exists() and not self.config.restart:
            print(f"[复用] 剧本质量总分: {output}", flush=True)
            return self._read_text(output)

        assert self.client is not None
        system = self._read_text(self.total_prompt)
        quality_keys = ("character", "hooks", "plot", "episode", "setting")
        sections = []
        titles = {task.key: task.title for task in self.tasks}
        for key in quality_keys:
            sections.append(f"# {titles[key]}评估报告\n\n{reports[key]}")
        user = (
            "以下提供五个基础剧本质量模块的完整评估报告。未提供小说/指令对照质量，"
            "必须采用‘无对照评估’权重。请严格按系统提示词完成计分，不要重新评估剧本。\n\n"
            + "\n\n---\n\n".join(sections)
        )
        report = self.client.text(system, user, label="剧本质量总分计分")
        self._atomic_write(output, report)
        return report

    @staticmethod
    def _extract_scores(text: str) -> dict[str, float]:
        scores: dict[str, float] = {}
        for label in SCORE_LABELS:
            pattern = rf"{re.escape(label)}\s*[：:]\s*\**\s*(\d+(?:\.\d+)?)\s*/\s*100"
            match = re.search(pattern, text)
            if match:
                scores[label] = float(match.group(1))
        return scores

    def _write_summary(self, reports: dict[str, str], quality_total: str, source_name: str) -> None:
        report_paths = {task.key: self.config.output_dir / task.output_name for task in self.tasks}
        all_scores: dict[str, float] = {}
        for key, report in reports.items():
            all_scores.update(self._extract_scores(report))
        all_scores.update(self._extract_scores(quality_total))

        payload = {
            "script": str(self.config.script_path.resolve()),
            "model": self.config.model,
            "evaluation_source": source_name,
            "scores": all_scores,
            "reports": {
                task.title: str(report_paths[task.key].resolve()) for task in self.tasks
            }
            | {"剧本质量总分": str((self.config.output_dir / "08_剧本质量总分计分.md").resolve())},
        }
        self._atomic_write(
            self.config.output_dir / "scores.json",
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

        display_rows = []
        for label in ("剧本创意总分", "剧本逻辑总分", "剧本质量最终总分"):
            value = all_scores.get(label)
            display_rows.append(f"| {label} | {value:.1f}/100 |" if value is not None else f"| {label} | 未解析到 |")
        detail_rows = []
        for task in self.tasks:
            relative = report_paths[task.key].name
            score_label = next((label for label in SCORE_LABELS if label.startswith(task.title)), "")
            value = all_scores.get(score_label) if score_label else None
            score = f"{value:.1f}/100" if value is not None else "见报告"
            detail_rows.append(f"| {task.title} | {score} | [{relative}]({relative}) |")
        detail_rows.append(
            "| 剧本质量总分计分 | "
            + (
                f"{all_scores['剧本质量最终总分']:.1f}/100"
                if "剧本质量最终总分" in all_scores
                else "见报告"
            )
            + " | [08_剧本质量总分计分.md](08_剧本质量总分计分.md) |"
        )
        summary = "\n".join(
            [
                "# 剧本评估总览",
                "",
                f"- 输入剧本：`{self.config.script_path.resolve()}`",
                f"- 评估模型：`{self.config.model}`",
                f"- 评估输入方式：{source_name}",
                "- 小说/IP改编对照：未启用",
                "",
                "## 顶层结果",
                "",
                "| 顶层维度 | 得分 |",
                "|---|---:|",
                *display_rows,
                "",
                "> 创意、逻辑与剧本质量采用不同量表，现有提示词未定义三者合并权重，因此不擅自计算单一总分。",
                "",
                "## 评估报告",
                "",
                "| 报告 | 得分 | 文件 |",
                "|---|---:|---|",
                *detail_rows,
            ]
        )
        self._atomic_write(self.config.output_dir / "00_评估总览.md", summary)

    def workload(self) -> dict[str, object]:
        script = self._read_text(self.config.script_path)
        use_evidence = self.config.context_mode == "evidence" or (
            self.config.context_mode == "auto" and len(script) > self.config.direct_char_limit
        )
        chunks = self._split_script(script, self.config.chunk_chars) if use_evidence else []
        return {
            "script": str(self.config.script_path.resolve()),
            "characters": len(script),
            "model": self.config.model,
            "output_dir": str(self.config.output_dir.resolve()),
            "context_mode": "evidence" if use_evidence else "direct",
            "evidence_chunks": len(chunks),
            "evaluation_reports": len(self.tasks),
            "quality_total_calls": 1,
            "estimated_model_calls": len(chunks) + len(self.tasks) + 1,
            "excluded": ["小说/指令对照质量"],
        }

    def run(self) -> Path:
        script = self._read_text(self.config.script_path)
        if not script:
            raise ValueError("剧本文件为空")
        self._prepare_output(self._manifest_data(script))
        self.client = OpenAIChatClient(
            model=self.config.model,
            max_output_tokens=self.config.max_output_tokens,
            timeout=self.config.timeout,
            retries=self.config.retries,
        )
        source_name, source = self._evaluation_source(script)
        reports = self._run_tasks(source_name, source)
        quality_total = self._quality_total(reports)
        self._write_summary(reports, quality_total, source_name)
        return self.config.output_dir / "00_评估总览.md"
