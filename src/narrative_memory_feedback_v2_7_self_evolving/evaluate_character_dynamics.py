from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .api_client import FakeLLMClient, OpenAICompatibleClient
from .io_utils import append_jsonl, index_jsonl, read_jsonl, stable_hash, utc_now, write_json
from .run_generation import call_and_record, extract_json_value
from .schema import ROOT, load_protocol

SCHEMA_VERSION = "1.0"
SUB_FIELDS = (
    "motivational_coherence",
    "agency_and_cost",
    "relationship_dynamics",
    "emotional_continuity",
    "character_integrity",
    "earned_evolution",
)
WEIGHTS = {
    "motivational_coherence": 0.20,
    "agency_and_cost": 0.15,
    "relationship_dynamics": 0.25,
    "emotional_continuity": 0.15,
    "character_integrity": 0.15,
    "earned_evolution": 0.10,
}
OPPORTUNITY_VALUES = {"NONE", "LATENT", "ACTIVE"}
KNOWN_FLAGS = {
    "MOTIVATION_GAP", "PLOT_DRIVEN_ACTION", "GOAL_DRIFT",
    "PLOT_PUPPET", "AGENCY_COLLAPSE", "COST_FREE_CHOICE", "FUNCTIONAL_SUPPORT_CHARACTER",
    "RELATIONSHIP_PROCEDURALIZATION", "TENSION_RESET", "INSTANT_RECONCILIATION",
    "ONE_SIDED_RELATIONSHIP", "DECLARED_NOT_DRAMATIZED", "DEBT_WITHOUT_RESPONSE",
    "EMOTIONAL_JUMP", "EMOTIONAL_RESET", "TOLD_EMOTION", "NO_EMOTIONAL_CONSEQUENCE",
    "OOC_ACTION", "KNOWLEDGE_LEAK", "ABILITY_JUMP", "BOUNDARY_VIOLATION", "INTERCHANGEABLE_CHARACTER",
    "UNEARNED_CHANGE", "DECLARED_GROWTH", "NO_STATE_GAIN", "ARC_RESET",
}


class CharacterDynamicsFakeClient(FakeLLMClient):
    def complete(self, messages, settings, *, seed, purpose):
        if purpose.startswith("character_dynamics"):
            return json.dumps({
                "subscores": {f: 82 for f in SUB_FIELDS},
                "relationship_opportunity": "ACTIVE",
                "relationship_analysis": {
                    "active_relationships": [], "tension_sources": [],
                    "reciprocal_pressure": True, "relationship_debt_present": True,
                    "relationship_debt_responded": True, "proceduralized_into_plan": False,
                },
                "flags": [],
                "details": {f: [{"type": "问题", "item": "假数据", "evidence": "测试", "points": -18}] for f in SUB_FIELDS},
            }, ensure_ascii=False)
        return super().complete(messages, settings, seed=seed, purpose=purpose)


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        import re
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _reconcile_from_details(value: dict[str, Any]) -> None:
    details = value.get("details")
    if not isinstance(details, dict):
        return
    for field in SUB_FIELDS:
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
            value["subscores"][field] = round(max(0.0, min(100.0, 100.0 + sum(points))), 2)


def validate_result(value: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("结果必须是对象")
    subscores = value.get("subscores")
    if not isinstance(subscores, dict):
        raise ValueError("subscores 必须是对象")
    opportunity = str(value.get("relationship_opportunity", "NONE")).upper()
    if opportunity not in OPPORTUNITY_VALUES:
        raise ValueError("relationship_opportunity 非法")
    details = value.get("details", {})
    if not isinstance(details, dict):
        details = {}
    relationship_analysis = value.get("relationship_analysis", {})
    if not isinstance(relationship_analysis, dict):
        relationship_analysis = {}
    flags = value.get("flags", [])
    if not isinstance(flags, list):
        flags = []
    cleaned_subscores: dict[str, float] = {}
    for field in SUB_FIELDS:
        score = _coerce_numeric(subscores.get(field))
        if score is None or not 0 <= score <= 100:
            raise ValueError(f"subscores.{field} 必须是 0-100 数值")
        cleaned_subscores[field] = score
    clean_details: dict[str, list[dict[str, Any]]] = {}
    for field in SUB_FIELDS:
        items = details.get(field, [])
        if not isinstance(items, list):
            raise ValueError(f"details.{field} 必须是数组")
        cleaned: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"details.{field} 元素必须是对象")
            cleaned.append({
                "type": str(item.get("type", "问题")),
                "item": str(item.get("item", "")),
                "evidence": str(item.get("evidence", "")),
                "points": item.get("points") if isinstance(item.get("points"), (int, float)) else 0,
            })
        clean_details[field] = cleaned
    result = {
        "subscores": cleaned_subscores,
        "relationship_opportunity": opportunity,
        "relationship_analysis": relationship_analysis,
        "flags": [str(x) for x in flags],
        "details": clean_details,
    }
    _reconcile_from_details(result)
    # CDQ 由代码计算
    result["cdq"] = round(sum(WEIGHTS[f] * result["subscores"][f] for f in SUB_FIELDS), 2)
    return result


def build_prompt(*, story: dict[str, Any], plan: dict[str, Any], previous_state: dict[str, Any], script: str) -> str:
    story_context = {
        "title": story["title"],
        "genre": story["genre"],
        "series_goal": story["series_goal"],
        "characters": [{"name": c["name"], "role": c["role"], "constraints": c.get("constraints", [])} for c in story["characters"]],
    }
    prev = {
        "character_state": previous_state.get("character_state", {}),
        "relationship_state": previous_state.get("relationship_state", {}),
        "current_goals": previous_state.get("current_goals", []),
    }
    return f"""你是以「人物是否像活人」为核心尺度的资深剧本评审。盲评：不得猜测实验条件。只评估人物动态质量，不评估文笔、不评估剧情逻辑。

故事与人物：
{json.dumps(story_context, ensure_ascii=False)}

上一集人物与关系状态：
{json.dumps(prev, ensure_ascii=False)}

当前集任务：
{json.dumps(plan.get('episode_goal', ''), ensure_ascii=False)}

当前集剧本：
{script}

对以下 6 个「人物动态」子项按扣分制打分（每项满分 100，从 100 起评，发现问题扣分、亮点加分，夹在 0-100，输出整数）：

1. motivational_coherence 动机与行动因果：人物为何行动，行动是否来自其目标/恐惧/责任/处境，而非作者硬塞。
   - 关键行动无动机来源 -12~20；动机只靠解释台词补 -6~12；为剧情突然改目标 -12~20；所有人自动服务集目标 -8~15。
2. agency_and_cost 主体性与选择代价：人物是否真的做选择并承担代价，而非接任务/查资料/交文件的剧情工具。
   - 只是接任务查资料交文件 -5~10；配角只提供信息 -8~15；有选择却不决策 -8~15；决策无风险/代价 -5~10；冲突被流程自动解决 -8~15。
3. relationship_dynamics 关系动态与张力：人物之间是否存在实质的目标冲突、情感亏欠、知识不对称、依赖、权力不平衡，且双方都能施加影响。
   - 关系债被直接改写为计划/任务（RELATIONSHIP_PROCEDURALIZATION）-12~20；伤害或隐瞒被一次谈话消除 -15~25；张力只在台词说明 -6~12；关系单向一方无主体性 -8~15；合作后旧矛盾无声重置 -10~18；普通流程争议冒充情感冲突 -5~10；关系变化无后续行为差异 -8~15。
4. emotional_continuity 情绪真实度与连续性：情绪是否有触发、惯性、残留，并通过行为体现。
   - 情绪突跳无触发 -10~18；重场后情绪归零 -8~15；情绪只靠旁白/总结说明 -5~10；所有人同一种情绪表达 -5~10；情绪不影响行为 -6~12。
5. character_integrity 人物完整性与边界：人设、能力、职业边界、知识归属、行为风格是否成立。
   - 明显违背人设 -15~25；无理由突破道德/职业边界 -12~20；使用不该知道的信息 -12~20；能力突增/突失 -10~18；换名后行为仍成立 -5~12；所有人同质化 -5~10。
6. earned_evolution 可信演化与人物增量：人物/关系变化是否经过触发—选择—代价—结果；不要求每集都变，无变化时需合理保留张力与状态。
   - 信任/立场/目标突翻 -12~20；只用一句话宣告成长 -8~15；关系变化无后果 -6~12；本该有后果却无增量 -6~12；同类变化反复重演归零 -8~15。

【评分校准】85 分以上必须在该子项 details 列出具体亮点及文本证据，否则不高于 84。无问题也无亮点落在 75-84。关系子项特别强调：吵架/意见不同不等于关系张力；只有双方都有利害、都能施加影响、并通过行动体现的才是真实张力。

【relationship_opportunity】判定本集是否存在自然的人际处理机会：
- NONE：本集任务没有自然的人际处理机会，不因关系没推进而扣分；
- LATENT：既有关系债/张力在场但本集只需保持存在；
- ACTIVE：人物直接接触了既有伤害/隐瞒/信任/权力/责任问题，若仍完全忽略必须扣关系分。

只输出合法 JSON，不要 Markdown：
{{"subscores":{{"motivational_coherence":0,"agency_and_cost":0,"relationship_dynamics":0,"emotional_continuity":0,"character_integrity":0,"earned_evolution":0}},"relationship_opportunity":"NONE|LATENT|ACTIVE","relationship_analysis":{{"active_relationships":[],"tension_sources":[],"reciprocal_pressure":true,"relationship_debt_present":false,"relationship_debt_responded":false,"proceduralized_into_plan":false}},"flags":[],"details":{{"motivational_coherence":[],"agency_and_cost":[],"relationship_dynamics":[],"emotional_continuity":[],"character_integrity":[],"earned_evolution":[]}}}}

- subscores 每个字段必须是 0-100 整数（不要带单位）；details 必须列出全部 6 个子项，每条含 type(问题/亮点)、item、evidence、points。
- flags 从以下取用：MOTIVATION_GAP, PLOT_DRIVEN_ACTION, GOAL_DRIFT, PLOT_PUPPET, AGENCY_COLLAPSE, COST_FREE_CHOICE, FUNCTIONAL_SUPPORT_CHARACTER, RELATIONSHIP_PROCEDURALIZATION, TENSION_RESET, INSTANT_RECONCILIATION, ONE_SIDED_RELATIONSHIP, DECLARED_NOT_DRAMATIZED, DEBT_WITHOUT_RESPONSE, EMOTIONAL_JUMP, EMOTIONAL_RESET, TOLD_EMOTION, NO_EMOTIONAL_CONSEQUENCE, OOC_ACTION, KNOWLEDGE_LEAK, ABILITY_JUMP, BOUNDARY_VIOLATION, INTERCHANGEABLE_CHARACTER, UNEARNED_CHANGE, DECLARED_GROWTH, NO_STATE_GAIN, ARC_RESET。
- relationship_analysis 的 proceduralized_into_plan 指：前集建立的是伤害/亏欠/隐瞒/猜忌/信任债，本集却直接改写成可执行计划/任务并在完成后视作关系已解决；relationship_debt_responded 指双方是否真的通过情感回应/让步/代价/重新划界来处理了该债务。
- 字符串值内禁止英文双引号和英文单引号；引用原话用中文引号「」。
- 只输出上述 JSON 对象，不要包在 result/output/data 字段。"""


def run(output_dir: Path, *, input_dir: Path | None = None, fake_api: bool = False, execute_api: bool = False, limit: int | None = None, story_filter: set[str] | None = None, run_filter: set[str] | None = None) -> int:
    config, stories, plans, _, _ = load_protocol()
    if not fake_api and not (execute_api and config["runtime"]["api_approved"]):
        raise RuntimeError("真实 API 被协议安全门阻止")
    client = CharacterDynamicsFakeClient() if fake_api else OpenAICompatibleClient.from_environment("evaluation")
    settings = dict(config["evaluation"])
    settings["temperature"] = 0.0
    data_dir = input_dir if input_dir is not None else output_dir
    stories_by_id = {item["story_id"]: item for item in stories}
    plans_by_key = {(item["story_id"], item["episode_id"]): item for item in plans}
    generations = read_jsonl(data_dir / "generations.jsonl")
    if story_filter:
        generations = [row for row in generations if row["story_id"] in story_filter]
    if run_filter:
        generations = [row for row in generations if row["run_id"] in run_filter]
    if limit is not None:
        generations = generations[:limit]
    states = index_jsonl(data_dir / "state_updates.jsonl", "job_id")
    existing = index_jsonl(output_dir / "character_dynamics.jsonl", "job_id")
    completed = 0
    skipped = 0
    for generation in generations:
        job_id = generation["job_id"]
        if job_id in existing:
            completed += 1
            continue
        story = stories_by_id[generation["story_id"]]
        plan = plans_by_key.get((generation["story_id"], generation["episode_id"])) or {}
        previous_state = story["initial_state"] if generation.get("previous_job_id") is None else states[generation["previous_job_id"]]["next_state"]
        prompt = build_prompt(story=story, plan=plan, previous_state=previous_state, script=generation["script"])
        result = None
        last_error = ""
        for attempt in range(4):
            retry = "" if attempt == 0 else f"\n\n上次输出未通过校验：{last_error}。重新输出完整合法 JSON。"
            raw = call_and_record(client, messages=[{"role": "user", "content": prompt + retry}], settings=settings, seed=generation["seed"] + 300 + attempt, purpose=f"character_dynamics_{attempt + 1}", job=generation, output_dir=output_dir)
            candidates = [raw]
            try:
                candidates.append(raw.replace("'", '"'))
            except Exception:
                pass
            for candidate in candidates:
                try:
                    result = validate_result(extract_json_value(candidate))
                    break
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    last_error = str(exc)
            if result is not None:
                break
        if result is None:
            skipped += 1
            continue
        append_jsonl(output_dir / "character_dynamics.jsonl", {
            "schema_version": SCHEMA_VERSION,
            "job_id": job_id,
            "trajectory_id": generation["trajectory_id"],
            "story_id": generation["story_id"],
            "run_id": generation["run_id"],
            "episode_id": generation["episode_id"],
            "condition": generation["condition"],
            "cdq": result["cdq"],
            "subscores": result["subscores"],
            "relationship_opportunity": result["relationship_opportunity"],
            "relationship_analysis": result["relationship_analysis"],
            "flags": result["flags"],
            "details": result["details"],
            "created_at": utc_now(),
        })
        existing[job_id] = True
        completed += 1
    print(f"[done] completed={completed} skipped={skipped}")
    return completed


def analyze(output_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(output_dir / "character_dynamics.jsonl")
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[row["trajectory_id"]].append(row)
    trajectories = []
    for trajectory_id, traj_rows in sorted(by_trajectory.items()):
        cdq = mean(r["cdq"] for r in traj_rows)
        subs = {f: mean(r["subscores"][f] for r in traj_rows) for f in SUB_FIELDS}
        flags = Counter(f for r in traj_rows for f in r["flags"])
        opportunities = Counter(r["relationship_opportunity"] for r in traj_rows)
        proc_count = sum(1 for r in traj_rows if r["relationship_analysis"].get("proceduralized_into_plan"))
        trajectories.append({
            "trajectory_id": trajectory_id,
            "story_id": traj_rows[0]["story_id"],
            "run_id": traj_rows[0]["run_id"],
            "condition": traj_rows[0]["condition"],
            "cdq": round(cdq, 4),
            "subscores": {f: round(subs[f], 4) for f in SUB_FIELDS},
            "relationship_proceduralization_count": proc_count,
            "flag_counts": dict(flags),
            "opportunity_distribution": dict(opportunities),
        })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "episode_count": len(rows),
        "trajectory_count": len(trajectories),
        "trajectories": trajectories,
    }
    write_json(output_dir / "character_dynamics_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--fake-api", action="store_true")
    parser.add_argument("--execute-api", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stories", type=str)
    parser.add_argument("--runs", type=str)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.analyze_only:
        analyze(args.output_dir)
        return
    story_filter = {x.strip() for x in args.stories.split(",") if x.strip()} if args.stories else None
    run_filter = {x.strip() for x in args.runs.split(",") if x.strip()} if args.runs else None
    count = run(args.output_dir, input_dir=args.input_dir, fake_api=args.fake_api, execute_api=args.execute_api, limit=args.limit, story_filter=story_filter, run_filter=run_filter)
    print(f"evaluated_jobs={count} fake_api={args.fake_api}")


if __name__ == "__main__":
    main()
