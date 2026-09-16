"""标注稳定性预检（执行计划第 1 步）。

目的：评估 narrative_signature 标注器对“同一剧本”重复标注的可复现性，
决定它能否作为重复的量化指标。

方法：
- 取 4 个 v6 条件目录 × 6 个故事，各抽 1 集（E05），共 24 集；
- 真实管线用 temperature=0.0，标签确定，两次同 seed 必然一致，无法暴露方差；
  因此本预检用 temperature=0.3 + 两个不同 seed 各标注一遍，以模型自身的
  采样方差作为“标注可靠性”的下界估计；
- 逐维度计算两遍一致率（单值字段精确匹配；mechanism_operations 额外报
  集合一致与 Jaccard；scene_functions 报 LCS 相似；expressive_motifs 报 Jaccard）；
- 并用 structured_similarity 给出两遍之间的核心结构相似度分布。

输出：runs/signature_stability_v1_1.json + 打印汇总表。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, pstdev

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # .../agent_memory_manage

from experience_following.settings import (  # noqa: E402
    EVALUATOR_API_KEY, EVALUATOR_BASE_URL, EVALUATOR_MODEL,
)
from narrative_memory_feedback_v2_2.api_client import OpenAICompatibleClient  # noqa: E402
from narrative_memory_feedback_v2_2.evaluate_signatures import _coerce_controlled  # noqa: E402
from narrative_memory_feedback_v2_2.narrative_signature import (  # noqa: E402
    SIGNATURE_SCHEMA_VERSION, build_signature_prompt, validate_signature,
    structured_similarity, _jaccard, _lcs_similarity,
)
from narrative_memory_feedback_v2_2.run_generation import extract_json_value  # noqa: E402

RDIR = ROOT / "runs"
RAW_OUT = RDIR / "signature_stability_raw_v1_1.jsonl"
COND_DIRS = {
    "A_state_only": "nmf_v3_multistory_v6",
    "C_high_homogeneous_frozen": "nmf_v3_multistory_v6_c",
    "D_high_diverse_append": "nmf_v3_multistory_v6_d",
    "E_high_homogeneous_append": "nmf_v3_multistory_v6_e",
}
STORIES = ["S01", "S02", "S03", "S04", "S05", "S06"]
EPISODE = "E05"
SAMPLE_TEMP = 0.3
MAX_TOKENS = 4096

CATEGORICAL = [
    "macro_move", "trigger_type", "obstacle_type", "state_change_target",
    "information_delta", "causal_relation", "turn_type", "resolution_type",
    "hook_type", "interaction_pattern", "relationship_change",
]
# 核心维度（结构化相似度里权重最高的几项，论文最依赖它们）
CORE_CATEGORICAL = ["mechanism_operations", "information_delta", "turn_type", "resolution_type"]

client = OpenAICompatibleClient(
    api_key=EVALUATOR_API_KEY, base_url=EVALUATOR_BASE_URL, model=EVALUATOR_MODEL,
)


def annotate(script: str, macro_move: str, move_evidence: str, seed: int) -> dict | None:
    prompt = build_signature_prompt(script=script, macro_move=macro_move, move_evidence=move_evidence)
    settings = {"temperature": SAMPLE_TEMP, "max_output_tokens": MAX_TOKENS, "top_p": 1.0}
    last_error = ""
    for attempt in range(5):
        retry = "" if attempt == 0 else f"\n\n上次结构未通过：{last_error}。重新输出完整 JSON，所有分类字段只能逐字使用上方受控值，不得自造近义标签；若不确定，使用 OTHER。"
        raw = client.complete(
            [{"role": "user", "content": prompt + retry}], settings,
            seed=seed + 200 + attempt, purpose=f"narrative_signature_{attempt + 1}",
        )
        try:
            cand = extract_json_value(raw)
            cand = _coerce_controlled(cand)
            validate_signature(cand, macro_move)
            return cand
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
    print(f"  !! 标注失败 seed={seed}: {last_error[:120]}")
    return None


def main() -> None:
    reuse = RAW_OUT.exists() and RAW_OUT.stat().st_size > 0
    if reuse:
        print(f"检测到 {RAW_OUT} 已存在，直接复用已有两遍标注，跳过 API 调用。")
        raw = [json.loads(l) for l in RAW_OUT.open(encoding="utf-8")]
        for r in raw:
            s1, s2 = r["signature_pass1"], r["signature_pass2"]
            agree = defaultdict(lambda: [0, 0])
            mech_jac, scene_lcs, motif_jac = [], [], []
            core_sims, decisive_match = [], []
            per_ep = []
            total, ok = len(raw), len(raw)
            for f in CATEGORICAL:
                agree[f][1] += 1
                if s1[f] == s2[f]:
                    agree[f][0] += 1
            mech_jac.append(_jaccard(s1["mechanism_operations"], s2["mechanism_operations"]))
            scene_lcs.append(_lcs_similarity(s1["scene_functions"], s2["scene_functions"]))
            motif_jac.append(_jaccard(s1["expressive_motifs"], s2["expressive_motifs"]))
            sim = structured_similarity(s1, s2)
            core_sims.append(sim["core_structural_similarity"])
            decisive_match.append(sim["decisive_change_match"])
            per_ep.append({
                "condition": r["condition"], "story": r["story"], "ok": True,
                "core_structural_similarity": sim["core_structural_similarity"],
                "repetition_class": sim["repetition_class"],
                "mechanism_operations_jaccard": round(mech_jac[-1], 4),
                "scene_functions_lcs": round(scene_lcs[-1], 4),
                "expressive_motifs_jaccard": round(motif_jac[-1], 4),
                "pass1_mechanism_operations": s1["mechanism_operations"],
                "pass2_mechanism_operations": s2["mechanism_operations"],
                "pass1_information_delta": s1["information_delta"],
                "pass2_information_delta": s2["information_delta"],
                "pass1_turn_type": s1["turn_type"],
                "pass2_turn_type": s2["turn_type"],
            })
        _emit(agree, mech_jac, scene_lcs, motif_jac, core_sims, decisive_match, per_ep,
              total, ok)
        return

    agree = defaultdict(lambda: [0, 0])              # field -> [match, total]
    mech_jac, scene_lcs, motif_jac = [], [], []
    core_sims, decisive_match = [], []
    per_ep = []
    total = 0
    ok = 0

    if RAW_OUT.exists():
        RAW_OUT.unlink()

    for cond, d in COND_DIRS.items():
        gens = {r["job_id"]: r for r in (json.loads(l) for l in open(RDIR / d / "generations.jsonl"))}
        exts = {r["job_id"]: r for r in (json.loads(l) for l in open(RDIR / d / "extractions.jsonl"))}
        for s in STORIES:
            gen = next((g for g in gens.values()
                        if g["story_id"] == s and g["episode_id"] == EPISODE and g["run_id"] == "R01"), None)
            if gen is None:
                continue
            total += 1
            ext = exts[gen["job_id"]]["result"]
            macro = ext["primary_move_id"]
            ev = ext.get("move_evidence", "")
            print(f"[{total:2}] {cond:28} {s}/E05  seed={gen['seed']}  script_len={len(gen['script'])}", flush=True)
            s1 = annotate(gen["script"], macro, ev, gen["seed"])
            s2 = annotate(gen["script"], macro, ev, gen["seed"] + 1)
            if s1 is None or s2 is None:
                per_ep.append({"condition": cond, "story": s, "ok": False, "error": "annotation_failed"})
                continue
            ok += 1

            for f in CATEGORICAL:
                agree[f][1] += 1
                if s1[f] == s2[f]:
                    agree[f][0] += 1
            mech_jac.append(_jaccard(s1["mechanism_operations"], s2["mechanism_operations"]))
            scene_lcs.append(_lcs_similarity(s1["scene_functions"], s2["scene_functions"]))
            motif_jac.append(_jaccard(s1["expressive_motifs"], s2["expressive_motifs"]))

            sim = structured_similarity(s1, s2)
            core_sims.append(sim["core_structural_similarity"])
            decisive_match.append(sim["decisive_change_match"])
            rec = {
                "condition": cond, "story": s, "ok": True,
                "core_structural_similarity": sim["core_structural_similarity"],
                "repetition_class": sim["repetition_class"],
                "mechanism_operations_jaccard": round(mech_jac[-1], 4),
                "scene_functions_lcs": round(scene_lcs[-1], 4),
                "expressive_motifs_jaccard": round(motif_jac[-1], 4),
                "pass1_mechanism_operations": s1["mechanism_operations"],
                "pass2_mechanism_operations": s2["mechanism_operations"],
                "pass1_information_delta": s1["information_delta"],
                "pass2_information_delta": s2["information_delta"],
                "pass1_turn_type": s1["turn_type"],
                "pass2_turn_type": s2["turn_type"],
                "signature_pass1": s1,
                "signature_pass2": s2,
            }
            per_ep.append(rec)
            with RAW_OUT.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"condition": cond, "story": s,
                                     "signature_pass1": s1, "signature_pass2": s2},
                                    ensure_ascii=False) + "\n")

    _emit(agree, mech_jac, scene_lcs, motif_jac, core_sims, decisive_match, per_ep, total, ok)


def _emit(agree, mech_jac, scene_lcs, motif_jac, core_sims, decisive_match, per_ep, total, ok) -> None:
    summary = {
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "sample_temperature": SAMPLE_TEMP,
        "episodes_selected": total,
        "episodes_labeled_ok": ok,
        "categorical_agreement": {f: {"match": v[0], "total": v[1], "rate": round(v[0] / v[1], 4)} for f, v in agree.items()},
        "core_dimension_agreement": {f: round(agree[f][0] / agree[f][1], 4) for f in CORE_CATEGORICAL if f in agree},
        "list_field_similarity": {
            "mechanism_operations_jaccard": round(mean(mech_jac), 4),
            "scene_functions_lcs": round(mean(scene_lcs), 4),
            "expressive_motifs_jaccard": round(mean(motif_jac), 4),
        },
        "between_pass_core_similarity": {
            "mean": round(mean(core_sims), 4) if core_sims else None,
            "median": round(median(core_sims), 4) if core_sims else None,
            "min": round(min(core_sims), 4) if core_sims else None,
            "sd": round(pstdev(core_sims), 4) if len(core_sims) > 1 else None,
            "decisive_change_match_rate": round(mean(decisive_match), 4) if decisive_match else None,
        },
        "per_episode": per_ep,
    }
    out = RDIR / "signature_stability_v1_1.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 标注稳定性预检结果 =====")
    print(f"采样温度={SAMPLE_TEMP}  选定集数={total}  成功标注={ok}")
    print("\n单值维度 两遍一致率：")
    for f in CATEGORICAL:
        m, t = agree[f]
        flag = "OK " if (m / t) >= 0.70 else "LOW"
        print(f"  [{flag}] {f:22} {m}/{t} = {m/t:.1%}")
    print("\n列表字段 两遍相似度（均值）：")
    print(f"  mechanism_operations Jaccard = {mean(mech_jac):.3f}")
    print(f"  scene_functions      LCS     = {mean(scene_lcs):.3f}")
    print(f"  expressive_motifs    Jaccard = {mean(motif_jac):.3f}")
    print("\n两遍之间核心结构相似度：")
    if core_sims:
        print(f"  mean={mean(core_sims):.3f}  median={median(core_sims):.3f}  min={min(core_sims):.3f}  sd={pstdev(core_sims):.3f}")
        print(f"  关键三连(information_delta/turn_type/resolution_type)全一致率 = {mean(decisive_match):.1%}")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
