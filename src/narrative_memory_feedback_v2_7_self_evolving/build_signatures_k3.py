"""k=3 多数投票式细粒度签名标注（执行计划第 2 步，A 条件先行）。

相对 evaluate_signatures 的单遍 temp=0 标注，本脚本对每集做 K=3 次独立
采样（temperature=0.3，不同 seed），再按维度做多数投票：
- 单值维度：3 次中取众数（2/3 即胜；3 全不同则回退 pass1）；
- 列表维度（mechanism_operations/scene_functions/expressive_motifs）：
  仅保留在 >=2 遍中出现的元素，按出现次序合并；
- 文本字段（mechanism_summary/evidence）：取自与聚合后 mechanism_operations
  集合重合度最高的那一遍。

产物文件名严格对齐 analyze_signatures 的读取约定
（narrative_signatures_v1_1.jsonl），可直接被下游分析复用。
支持断点续跑：已写入输出文件的 job_id 会被跳过。

用法：
  python build_signatures_k3.py            # 默认仅 A（R01）
  python build_signatures_k3.py A C D E    # 指定条件
  python build_signatures_k3.py --workers 8 A
  # 标注 R02（目录带 _r02 后缀，只取 run_id=R02 的集）
  python build_signatures_k3.py --run-id R02 --dir-suffix _r02 A_state_only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # .../agent_memory_manage

# 真实评测凭证（环境无 env 时回退到 experience_following.settings）
try:
    from experience_following.settings import (  # noqa: E402
        EVALUATOR_API_KEY, EVALUATOR_BASE_URL, EVALUATOR_MODEL,
    )
except Exception:  # noqa: BLE001
    EVALUATOR_API_KEY = os.environ["NMF_EVALUATION_API_KEY"]
    EVALUATOR_BASE_URL = os.environ["NMF_EVALUATION_BASE_URL"]
    EVALUATOR_MODEL = os.environ["NMF_EVALUATION_MODEL"]

from narrative_memory_feedback_v2_2.api_client import OpenAICompatibleClient  # noqa: E402
from narrative_memory_feedback_v2_2.evaluate_signatures import _coerce_controlled  # noqa: E402
from narrative_memory_feedback_v2_2.io_utils import stable_hash, utc_now  # noqa: E402
from narrative_memory_feedback_v2_2.narrative_signature import (  # noqa: E402
    SIGNATURE_SCHEMA_VERSION, build_signature_prompt, validate_signature,
)
from narrative_memory_feedback_v2_2.run_generation import extract_json_value  # noqa: E402

RDIR = ROOT / "runs"
COND_DIRS = {
    "A_state_only": "nmf_v3_multistory_v6",
    "B_high_diverse_frozen": "nmf_v3_multistory_v6_b",
    "C_high_homogeneous_frozen": "nmf_v3_multistory_v6_c",
    "D_high_diverse_append": "nmf_v3_multistory_v6_d",
    "E_high_homogeneous_append": "nmf_v3_multistory_v6_e",
    "F_low_append": "nmf_v3_multistory_v6_f",
    "G_prev_only": "nmf_v3_multistory_v6_g",
}
K = 3
SAMPLE_TEMP = 0.3
MAX_TOKENS = 4096
CATEGORICAL = [
    "macro_move", "trigger_type", "obstacle_type", "state_change_target",
    "information_delta", "causal_relation", "turn_type", "resolution_type",
    "hook_type", "interaction_pattern", "relationship_change",
]
LIST_FIELDS = ["mechanism_operations", "scene_functions", "expressive_motifs"]
# 核心维度（不稳定，最依赖多数投票）
CORE_DIMS = ["information_delta", "turn_type", "resolution_type"]

_lock = threading.Lock()
_write_lock = threading.Lock()
client = OpenAICompatibleClient(api_key=EVALUATOR_API_KEY, base_url=EVALUATOR_BASE_URL, model=EVALUATOR_MODEL)


def annotate_one(script: str, macro_move: str, move_evidence: str, seed: int):
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
    return None


def _majority(vals):
    c = Counter(vals)
    items = c.most_common()
    if len(items) == 1:
        return items[0][0]
    if items[0][1] > items[1][1]:  # 2/3
        return items[0][0]
    return vals[0]  # 3 全不同 -> pass1


def _majority_list(lists):
    cnt = Counter()
    for L in lists:
        for x in set(L):
            cnt[x] += 1
    kept = [x for x in lists[0] if cnt[x] >= 2]
    seen = set(kept)
    for L in lists[1:]:
        for x in L:
            if cnt[x] >= 2 and x not in seen:
                kept.append(x)
                seen.add(x)
    if not kept:
        # 三遍无任一项达到 2/3 多数：回退到 pass1（已逐条校验，必为非空合法列表）
        return list(lists[0]) if lists[0] else []
    return kept


def _jac(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def aggregate(passes):
    if len(passes) == 1:
        return passes[0]
    agg = {f: _majority([p[f] for p in passes]) for f in CATEGORICAL}
    for f in LIST_FIELDS:
        agg[f] = _majority_list([p[f] for p in passes])
    best = max(range(len(passes)), key=lambda i: _jac(passes[i]["mechanism_operations"], agg["mechanism_operations"]))
    agg["mechanism_summary"] = passes[best]["mechanism_summary"]
    agg["evidence"] = passes[best]["evidence"]
    return agg


def process_condition(cond: str, d: str, workers: int, run_id: str = "R01") -> dict:
    run_dir = RDIR / d
    out_path = run_dir / f"narrative_signatures_v1_1.jsonl"
    raw_path = run_dir / "signature_k3_raw.jsonl"

    gens = [json.loads(l) for l in open(run_dir / "generations.jsonl")]
    gens = [g for g in gens if g.get("run_id") == run_id and "__probe" not in g["job_id"]]
    exts = {r["job_id"]: r for r in (json.loads(l) for l in open(run_dir / "extractions.jsonl"))}

    done = set()
    if out_path.exists():
        for l in open(out_path, encoding="utf-8"):
            done.add(json.loads(l)["job_id"])
    todo = [g for g in gens if g["job_id"] not in done]
    print(f"[{cond}] dir={d} run_id={run_id} total={len(gens)} "
          f"already_done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return {"condition": cond, "todo": 0}

    # 任务派发：每集 K 遍，全部并发
    tasks = {}
    pending = {}  # job_id -> {"gen":.., "received":0, "passes":{}}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for g in todo:
            ext = exts[g["job_id"]]["result"]
            macro = ext["primary_move_id"]
            ev = ext.get("move_evidence", "")
            pending[g["job_id"]] = {"gen": g, "received": 0, "passes": {}}
            for kk in range(K):
                fut = ex.submit(annotate_one, g["script"], macro, ev, g["seed"] + kk)
                tasks[fut] = (g["job_id"], kk)
        for fut in as_completed(tasks):
            job_id, kk = tasks[fut]
            try:
                sig = fut.result()
            except Exception as exc:  # noqa: BLE001
                sig = None
                print(f"  !! {job_id} pass{kk} exception: {str(exc)[:100]}")
            with _lock:
                st = pending[job_id]
                st["passes"][kk] = sig
                st["received"] += 1
                if st["received"] == K:
                    _finalize(st, cond, exts, out_path, raw_path)

    # 汇总质量指标（仅本批次）
    quality = []
    for job_id, st in pending.items():
        passes = [st["passes"][kk] for kk in range(K)]
        valid = [p for p in passes if p is not None]
        if not valid:
            quality.append({"job_id": job_id, "ok": False})
            continue
        # 核心维度在有效遍间的一致情况
        core_all_agree = all(
            len(set(p[dim] for p in valid)) <= 1 for dim in CORE_DIMS
        )
        quality.append({
            "job_id": job_id, "ok": True, "n_valid": len(valid),
            "core_dims_all_agree": core_all_agree,
        })
    n_q = len(quality)
    n_ok = sum(1 for q in quality if q["ok"])
    n_core_stable = sum(1 for q in quality if q.get("core_dims_all_agree"))
    print(f"[{cond}] 完成: 成功标注 {n_ok}/{n_q} 集; "
          f"核心三维度(3遍)全一致率 = {n_core_stable/n_q:.1%}" if n_q else f"[{cond}] 无任务")
    return {
        "condition": cond, "todo": len(todo), "ok": n_ok,
        "core_dims_all_agree_rate": round(n_core_stable / n_q, 4) if n_q else None,
    }


def _finalize(st, cond, exts, out_path, raw_path):
    g = st["gen"]
    passes = [st["passes"][kk] for kk in range(K)]
    valid = [p for p in passes if p is not None]
    job_id = g["job_id"]
    if not valid:
        return
    agg = aggregate(valid)
    try:
        validate_signature(agg, exts[job_id]["result"]["primary_move_id"])
    except Exception as exc:  # noqa: BLE001
        print(f"  !! 聚合签名校验失败 {job_id}: {exc}")
        return
    rec = {
        "signature_id": stable_hash({"job_id": job_id, "schema": SIGNATURE_SCHEMA_VERSION, "result": agg}),
        "schema_version": SIGNATURE_SCHEMA_VERSION,
        "job_id": job_id,
        "trajectory_id": g["trajectory_id"],
        "story_id": g["story_id"],
        "condition": cond,
        "episode_id": g["episode_id"],
        "signature": agg,
        "k_valid_passes": len(valid),
        "created_at": utc_now(),
    }
    with _write_lock:
        with out_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with raw_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"job_id": job_id, "condition": cond,
                                 "passes": [p for p in passes]}, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("conditions", nargs="*", default=["A_state_only"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--run-id", default="R01",
                    help="只标注该 run_id 的集（R01 / R02 ...）")
    ap.add_argument("--dir-suffix", default="",
                    help="run 目录名后缀，例如 _r02 指向 nmf_v3_multistory_v6_r02")
    args = ap.parse_args()
    os.environ.setdefault("NMF_API_TIMEOUT_SECONDS", "180")
    selected = {c: COND_DIRS[c] + args.dir_suffix for c in args.conditions if c in COND_DIRS}
    if not selected:
        print("可用条件:", list(COND_DIRS))
        return
    missing = [d for d in selected.values() if not (RDIR / d / "generations.jsonl").exists()]
    if missing:
        print("以下 run 目录缺少 generations.jsonl，请检查 --dir-suffix：", missing)
        return
    summary = []
    for cond, d in selected.items():
        summary.append(process_condition(cond, d, args.workers, run_id=args.run_id))
    print("\n===== k=3 标注汇总 =====")
    for s in summary:
        print(f"  {s['condition']:28} todo={s.get('todo')} ok={s.get('ok')} "
              f"core_dims_all_agree_rate={s.get('core_dims_all_agree_rate')}")


if __name__ == "__main__":
    main()
