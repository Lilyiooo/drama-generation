"""多 Agent 剧本评测 Web 服务。

- 上传 txt 剧本 + 指定评测模型 -> 后台运行评测 pipeline
- 前端轮询进度 -> 完成后读取 multi_agent_result.json 返回结构化结果
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from drama_evaluator import MultiAgentEvaluationConfig, MultiAgentEvaluationPipeline
from drama_evaluator.multi_agent import DIMENSIONS, LABELS, UNITS

BASE = Path(__file__).resolve().parent


def load_local_env() -> None:
    """与 evaluate_multi_agent.py 相同的 .env 加载逻辑。"""
    env_path = BASE / ".env"
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


load_local_env()

app = Flask(__name__, static_folder="static", static_url_path="/static")

JOBS_DIR = BASE / "web_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

TOTAL_REVIEWS = 9
TOTAL_AUDITS = 3
TOTAL_ARBITRATIONS = 3


# ---------------------------------------------------------------------------
# 后台任务
# ---------------------------------------------------------------------------
def _run_job(job_id: str, script_path: Path, model: str, output_dir: Path) -> None:
    try:
        config = MultiAgentEvaluationConfig(
            script_path=script_path,
            output_dir=output_dir,
            model=model,
            review_workers=3,
            audit_workers=3,
            arbitration_workers=3,
        )
        pipeline = MultiAgentEvaluationPipeline(config)
        pipeline.run()
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["ended_at"] = datetime.now().isoformat(timespec="seconds")
    except Exception as error:  # noqa: BLE001 - 上报给前端展示
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = f"{type(error).__name__}: {error}"
            _jobs[job_id]["ended_at"] = datetime.now().isoformat(timespec="seconds")


def _compute_progress(job: dict[str, Any]) -> dict[str, Any]:
    """按输出目录里已写出的评审/审核/仲裁产物估算进度。"""
    out = Path(job["output_dir"])
    reviews = audits = arbitrations = 0
    try:
        reviews = len(list((out / "multi_agent" / "reviews").rglob("*.json")))
        audits = len(list((out / "multi_agent" / "audits").glob("*.json")))
        arbitrations = len(list((out / "multi_agent" / "arbitrations").glob("*.json")))
    except OSError:
        pass
    reviews = min(reviews, TOTAL_REVIEWS)
    audits = min(audits, TOTAL_AUDITS)
    arbitrations = min(arbitrations, TOTAL_ARBITRATIONS)

    if reviews < TOTAL_REVIEWS:
        stage = f"独立评审 {reviews}/{TOTAL_REVIEWS}"
    elif audits < TOTAL_AUDITS:
        stage = f"台账审核 {audits}/{TOTAL_AUDITS}"
    else:
        stage = f"争议仲裁/汇总 {arbitrations}/{TOTAL_ARBITRATIONS}"

    done = reviews + audits + arbitrations
    total = TOTAL_REVIEWS + TOTAL_AUDITS + TOTAL_ARBITRATIONS
    percent = round(done / total * 100)
    return {
        "stage": stage,
        "reviews": reviews,
        "audits": audits,
        "arbitrations": arbitrations,
        "percent": min(percent, 99),
    }


def _quote_text(evidence: dict[str, Any]) -> str:
    evidence = evidence or {}
    episode = evidence.get("episode", "")
    scene = evidence.get("scene", "")
    quote = evidence.get("quote", "")
    contrast = evidence.get("contrast_quote", "")
    text = ""
    if quote:
        loc = f"[{episode}/{scene}]" if (episode or scene) else ""
        text = f"{loc} {quote}".strip()
    if contrast:
        cep = evidence.get("contrast_episode", "")
        csc = evidence.get("contrast_scene", "")
        cloc = f"[{cep}/{csc}]" if (cep or csc) else ""
        text += f"  ⇄  {cloc} {contrast}".rstrip()
    return text


def _build_result(output_dir: Path) -> dict[str, Any]:
    raw = json.loads((output_dir / "multi_agent_result.json").read_text(encoding="utf-8"))
    dims = raw.get("dimensions", {})

    entries_by_dim: dict[str, list[dict[str, Any]]] = {}
    for entry in raw.get("final_ledger_entries", []):
        entries_by_dim.setdefault(entry.get("dimension", ""), []).append(entry)

    dimensions: dict[str, Any] = {}
    for key in DIMENSIONS:
        dim = dims.get(key, {})
        unit_scores = dim.get("unit_ledger_scores", {})
        units = [
            {
                "key": unit_key,
                "label": label,
                "score": unit_scores.get(unit_key),
                "weight": round(weight * 100),
            }
            for unit_key, (label, weight) in UNITS[key].items()
        ]
        entries = []
        for entry in entries_by_dim.get(key, []):
            unit_key = entry.get("unit", "")
            entries.append({
                "id": entry.get("audit_id", ""),
                "unit": UNITS[key].get(unit_key, ("", 0.0))[0],
                "kind": entry.get("kind"),
                "severity": entry.get("severity"),
                "points": entry.get("points"),
                "status": entry.get("status"),
                "claim": entry.get("claim", ""),
                "reason": entry.get("final_reason", ""),
                "quote": _quote_text(entry.get("evidence", {})),
            })
        dimensions[key] = {
            "key": key,
            "label": dim.get("label") or LABELS[key],
            "fusion_score": dim.get("fusion_score"),
            "fusion_rating": dim.get("fusion_rating"),
            "initial_mean": dim.get("initial_mean_score"),
            "ledger_score": dim.get("ledger_score"),
            "ledger_rating": dim.get("rating"),
            "reviewer_scores": dim.get("reviewer_scores", {}),
            "confirmed_count": dim.get("confirmed_entry_count"),
            "rejected_count": dim.get("rejected_entry_count"),
            "units": units,
            "entries": entries,
        }

    return {
        "model": raw.get("model"),
        "script": raw.get("script"),
        "dimensions": dimensions,
        "suggestions": raw.get("next_stage_suggestions", []),
    }


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.route("/")
def index() -> str:
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/evaluate", methods=["POST"])
def evaluate() -> Any:
    file = request.files.get("file")
    model = (request.form.get("model") or "").strip() or "gpt-5.6-sol"
    if file is None or not file.filename:
        return jsonify({"error": "请上传 txt 剧本文档"}), 400
    if not file.filename.lower().endswith((".txt", ".md")):
        return jsonify({"error": "仅支持 .txt 或 .md 文本文件"}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    script_path = job_dir / "script.txt"
    output_dir = job_dir / "output"
    file.save(script_path)

    if not script_path.read_text(encoding="utf-8-sig", errors="ignore").strip():
        return jsonify({"error": "上传的剧本文档为空"}), 400

    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "model": model,
            "filename": file.filename,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": None,
            "output_dir": str(output_dir),
            "error": None,
        }

    thread = threading.Thread(
        target=_run_job, args=(job_id, script_path, model, output_dir), daemon=True
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>")
def job_status(job_id: str) -> Any:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在"}), 404

    payload = dict(job)
    payload["progress"] = _compute_progress(job) if job["status"] == "running" else None
    if job["status"] == "done":
        try:
            payload["result"] = _build_result(Path(job["output_dir"]))
        except Exception as error:  # noqa: BLE001
            payload["status"] = "error"
            payload["error"] = f"结果读取失败：{error}"
    return jsonify(payload)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
